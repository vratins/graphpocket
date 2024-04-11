#!/usr/bin/env python
import argparse
import pickle
import shutil
import sys
import time
import yaml
import os
from typing import Union

import numpy as np
import dgl
from dgl.dataloading import GraphDataLoader
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_scatter import scatter_mean
import torch.nn.functional as F


from sklearn.metrics import roc_curve, auc

import wandb
from tqdm import tqdm

from dataloader import get_dataloader, create_dataset
from graphpocket import GraphPocket
from model import ReceptorEncoderGVP, con_loss, get_batch_idx

sys.path.append('~/graphpocket/gnn')

def get_args():
    parser_config = argparse.ArgumentParser()

    parser_config.add_argument('--resume',
                            required=False,
                            default=False,
                            type=bool,
                            help='True if you have a model saved in the result directory')

    parser_config.add_argument('--config',
                        required=False,
                        default=None,
                        type=str)

    config_true, _ = parser_config.parse_known_args()
    if config_true.config:
        return parser_config.parse_args()
    else:

        parser = argparse.ArgumentParser()
        
        parser.add_argument('--resume',
                            required=False,
                            default=False,
                            type=bool,
                            help='True if you have a model saved in the result directory')
        
        #directories
        parser.add_argument('--pockets',required=False,default='~/dataset_graph/data',type=str)
        parser.add_argument('--pos_list',required=False,default='~/dataset_graph/TOUGH-M1/TOUGH-M1_positive.list',type=str)
        parser.add_argument('--neg_list',required=False,default='~/dataset_graph/TOUGH-M1/TOUGH-M1_negative.list',type=str)
        parser.add_argument('--cluster_map',required=False,default='~/graphpocket/cluster_map.pkl',type=str)
        parser.add_argument('--result_dir',required=False,default='~/graphpocket/results' ,type=str)
    
        #graph construction
        parser.add_argument('--knn_k',required=False,default=10,type=int)
        parser.add_argument('--algorithm',required=False,default='bruteforce-blas',type=str)
    
        #model config
        parser.add_argument('--input_scalar_size', required=False, default=24, type=int, help='Input scalar size')
        parser.add_argument('--output_scalar_size', required=False, default=64, type=int, help='Output scalar size')
        parser.add_argument('--edge_feat_size', required=False, default=1, type=int, help='Edge feature size')
        parser.add_argument('--vector_size', required=False, default=16, type=int, help='Vector size')
        parser.add_argument('--n_message_gvps', required=False, default=1, type=int)
        parser.add_argument('--n_update_gvps', required=False, default=1, type=int)
        parser.add_argument('--n_convs', required=False, default=3, type=int, help='Number of convolutions')
        parser.add_argument('--dropout', required=False, default=0.25, type=float, help='Dropout rate')
        parser.add_argument('--message_norm', required=False, default=24, help='Message Norm')
        parser.add_argument('--vector_gating', required=False, default=True, type=bool, help='Enable vector gating')
        parser.add_argument('--xavier_init', required=False, default=True, type=bool, help='Enable Xavier initialization')
    
    
        #loss margin and config
        parser.add_argument('--loss_margin',required=False,default=1.0,type=float)
    
        #train arguments
        parser.add_argument('--n_epochs', required=False, default=100, type=int, help='Number of epochs')
        parser.add_argument('--batch_size', required=False, default=64, type=int, help='Batch size')
        parser.add_argument('--n_workers', required=False, default=6, type=int, help='Number of workers')
        
        #optimizer parameters
        parser.add_argument('--optimizer_type', required=False, default='Adam', type=str, help='Optimizer type')
        parser.add_argument('--lr', required=False, default=0.001, type=float, help='Learning rate')
        parser.add_argument('--weight_decay', required=False, default=0.00001, type=float, help='Weight decay')
        
        #scheduler parameters
        parser.add_argument('--scheduler_type', required=False, default="StepLR", type=str, help='Scheduler type')
        parser.add_argument('--step_size', required=False, default=20, type=int, help='Step size for StepLR scheduler')
        parser.add_argument('--gamma', required=False, default=0.1, type=float, help='Gamma for StepLR scheduler')
    
        return parser.parse_args()

def main():

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'config' in args:
        with open('graphpocket/config/config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        #read in directories from config
        pocket_dir = os.path.expanduser(config['directories']['pockets'])
        pos_list = os.path.expanduser(config['directories']['pos_list'])
        neg_list = os.path.expanduser(config['directories']['neg_list'])
        seq_cluster_map = os.path.expanduser(config['directories']['cluster_map'])
        result_dir = os.path.expanduser(config['directories']['result_dir'])

        #train params
        train_config = config['train']
        batch_size, n_workers, epochs = train_config['batch_size'], train_config['n_workers'], train_config['n_epochs']

        #model params
        loss_margin = config['loss']['margin']
        model_params = config['model']
        in_scalar_size = model_params['input_scalar_size']
        out_scalar_size = model_params['output_scalar_size']
        edge_feat_size = model_params['edge_feat_size']
        n_message_gvps = model_params['n_message_gvps']
        n_update_gvps = model_params['n_update_gvps']
        vector_size = model_params['vector_size']
        message_norm = model_params['message_norm']
        n_convs = model_params['n_convs']
        dropout = model_params['dropout']

        opt_config = config['train']['optimizer']
        opt_type = opt_config['type']
        lr=opt_config['lr']
        weight_decay=opt_config['weight_decay']

        sched_config = config['train']['scheduler']
        scheduler_type = sched_config['type']
        params = sched_config['params']
        step_size=params['step_size']
        gamma=params['gamma']

        knn_k = config['graph']['threshold_k']
        algorithm = config['graph']['algorithm']

    else:
        
        pocket_dir = os.path.expanduser(args.pockets)
        pos_list = os.path.expanduser(args.pos_list)
        neg_list = os.path.expanduser(args.neg_list)
        seq_cluster_map = os.path.expanduser(args.cluster_map)
        result_dir = os.path.expanduser(args.result_dir)

        #train params
        batch_size = args.batch_size
        n_workers = args.n_workers
        epochs = args.n_epochs

        #model params
        loss_margin = args.loss_margin
        in_scalar_size= args.input_scalar_size
        out_scalar_size= args.output_scalar_size
        vector_size= args.vector_size
        n_message_gvps = args.n_message_gvps
        n_update_gvps = args.n_update_gvps
        edge_feat_size = args.edge_feat_size
        n_convs= args.n_convs
        dropout= args.dropout
        message_norm = args.message_norm

        opt_type = args.optimizer_type
        lr= args.lr
        weight_decay= args.weight_decay

        scheduler_type = args.scheduler_type
        step_size= args.step_size
        gamma= args.gamma

        knn_k = args.knn_k
        algorithm = args.algorithm

        loss_margin = args.loss_margin
        

    #set random seed manually
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    def get_optimizer(model_parameters, opt_type, lr, weight_decay):
        if opt_type == 'Adam':
            return Adam(model_parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

    def get_scheduler(optimizer, scheduler_type, step_size, gamma):
        if scheduler_type == "StepLR":
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
    if args.resume:
        if not os.path.exists(saved_model_path):
            raise FileNotFoundError(f"The saved model file was not found at {os.path.join(result_dir, 'model.pth.tar')}")
        
        state_dict = torch.load(saved_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    else:
        model = ReceptorEncoderGVP(
        in_scalar_size = in_scalar_size,
        out_scalar_size = out_scalar_size,
        edge_feat_size = edge_feat_size,
        n_message_gvps = n_message_gvps,
        n_update_gvps = n_update_gvps,
        vector_size = vector_size,
        n_convs = n_convs,
        dropout = dropout,
        message_norm = float(message_norm)
    )
        
    model.to(device)

    optimizer = get_optimizer(model.parameters(), opt_type, lr, weight_decay)
    scheduler = get_scheduler(optimizer, scheduler_type, step_size, gamma)

    #wandb
    wandb.init(project="graphpocket", config=config)

    #model_params
    print("Created model, now reading pockets into graphs..")

    if os.path.exists(os.path.join(pocket_dir, 'train_dataset1.pkl')):
        with open(os.path.join(pocket_dir, 'train_dataset.pkl'), 'rb') as f:
            train_dataset = pickle.load(f)
        with open(os.path.join(pocket_dir, 'test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)

    else:
        train_dataset, test_dataset = create_dataset(pos_list, neg_list, pocket_dir, seq_cluster_map, 
                                                     knn_k, algorithm, fold_nr=0, split_type='random')
        with open(os.path.join(pocket_dir, 'train_dataset.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(pocket_dir, 'test_dataset.pkl'), 'wb') as f:
            pickle.dump(test_dataset, f)
        
    print("Creating dataloaders..")
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)

    print("Size of train dataset: ", len(train_dataloader)*batch_size, "pairs")
    print("Size of test dataset: ", len(test_dataloader)*batch_size, "pairs") 

    #train func 
    def train(epoch):
        model.train()
        losses, pos_dists, neg_dists = [], [], []    
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Train Epoch: {epoch+1}")
        
        for batch_idx, ((graph1, graph2), label) in progress_bar:
            # torch.cuda.reset_peak_memory_stats(device)  
            graph1, graph2, label = graph1.to(device), graph2.to(device), label.to(device)
        
            batch_indx1 = get_batch_idx(graph1).to(device)
            batch_indx2 = get_batch_idx(graph2).to(device)
            optimizer.zero_grad()
                
            output1 = model(graph1, batch_indx1)
            output2 = model(graph2, batch_indx2)
                
            loss, pos_dist, neg_dist = con_loss(output1, output2, label, loss_margin)
    
            losses.append(loss.item())
            pos_dists.extend(pos_dist.cpu().numpy().tolist())
            neg_dists.extend(neg_dist.cpu().numpy().tolist())  
        
            loss.backward()
            optimizer.step()

            # print((torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()), torch.cuda.max_memory_allocated()/s)
                            
        return {'loss' : np.mean(losses), 'pos_dist' : np.mean(pos_dists), 'neg_dist' : np.mean(neg_dists)}

    #test func
    def test(epoch):
        model.eval()        
        all_dists, all_labels = [], []

        progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Test Epoch: {epoch+1}")

        with torch.no_grad():
            for batch_idx, ((graph1, graph2), label) in progress_bar:
                # torch.cuda.reset_peak_memory_stats(device)  
                graph1, graph2, label = graph1.to(device), graph2.to(device), label.to(device)

                batch_indx1 = get_batch_idx(graph1).to(device)
                batch_indx2 = get_batch_idx(graph2).to(device)
                
                output1 = model(graph1, batch_indx1)
                output2 = model(graph2, batch_indx2)
                
                dists = F.pairwise_distance(output1, output2).view(-1)
                all_dists.extend(dists.cpu().numpy())
                all_labels.extend(label.cpu().numpy())


        all_dists = np.array(all_dists)
        all_labels = np.array(all_labels)
        
        fpr, tpr, _ = roc_curve(1-all_labels, all_dists) #1-labels as higher distances is technically good for label 0 and roc wants probs
        roc_auc = auc(fpr, tpr)

        return {'AUC' : roc_auc}

    #epoch loop

    for epoch in range(epochs):  
        print("starting training runs")
        train_metrics = train(epoch)
        wandb.log({'train_loss': train_metrics['loss'], 
                   'train_pos_dist': train_metrics['pos_dist'], 
                   'train_neg_dist': train_metrics['neg_dist'],
                   'epoch': epoch+1})
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_metrics['loss']:.4f}")


        test_metrics = test(epoch)
        wandb.log({'AUC of ROC Curve': test_metrics['AUC'],
                  'epoch': epoch+1})
        print(f"Epoch {epoch+1}/{epochs}, Test AUC: {test_metrics['AUC']:.4f}")

        scheduler.step() 

        current_lr = scheduler.get_last_lr()
        print(f"Current Learning Rate: {current_lr}")
  
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
            os.path.join(result_dir, 'model.pth.tar'))
        print(f"Model saved at epoch {epoch+1}")

if __name__=='__main__':
    main()

