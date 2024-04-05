#!/usr/bin/env python
import argparse
import pickle
import shutil
import sys
import time
import yaml
import os

import numpy as np
import dgl
from dgl.dataloading import GraphDataLoader
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch_scatter import scatter_mean

import wandb
from tqdm import tqdm

from dataloader import get_dataloader, create_dataset
from graphpocket import GraphPocket
from model import ReceptorEncoderGVP, con_loss, get_batch_idx

sys.path.append('~/graphpocket/gnn')

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-resume',
                        required=False,
                        default=False,
                        type=bool,
                        help='True if you have a model saved in the result directory')

    return parser.parse_args()

def main():

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('graphpocket/config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #set random seed manually
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    #create train and test datasets [log the paths in config] -- make all from config?
    pocket_dir = os.path.expanduser(config['directories']['pockets'])
    pos_list = os.path.expanduser(config['directories']['pos_list'])
    neg_list = os.path.expanduser(config['directories']['neg_list'])
    seq_cluster_map = os.path.expanduser(config['directories']['cluster_map'])
    result_dir = os.path.expanduser(config['directories']['result_dir'])

    #model and train parameters
    train_config = config['train']
    batch_size, n_workers, epochs = train_config['batch_size'], train_config['n_workers'], train_config['n_epochs']

    def get_optimizer(model_parameters, optimizer_config):
        if optimizer_config['type'] == 'Adam':
            return Adam(model_parameters, lr=optimizer_config['lr'], weight_decay=optimizer_config['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")

    def get_scheduler(optimizer, scheduler_config):
        if scheduler_config['type'] == "StepLR":
            params = scheduler_config['params']
            return StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
        
    loss_margin = config['loss']['margin']

    def initialize_model(model_params, resume=False, saved_model_path=os.path.join(result_dir, 'model.pth.tar')):
        model = ReceptorEncoderGVP(
            in_scalar_size=model_params['input_scalar_size'],
            out_scalar_size=model_params['output_scalar_size'],
            vector_size=model_params['vector_size'],
            n_convs=model_params['n_convs'],
            dropout=model_params['dropout'],
        )
        
        if resume:
            if not os.path.exists(saved_model_path):
                raise FileNotFoundError(f"The saved model file was not found at {saved_model_path}")
            
            state_dict = torch.load(saved_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        
        return model

    model_config = config['model']
    model = initialize_model(model_config, args.resume)

    model.to(device)

    opt_config = config['train']['optimizer']
    optimizer = get_optimizer(model.parameters(), opt_config)

    sched_config = config['train']['scheduler']
    scheduler = get_scheduler(optimizer, sched_config)

    #wandb
    wandb.init(project="graphpocket", config=config)

    print("Created model, now reading pockets into graphs..")

    if os.path.exists(os.path.join(pocket_dir, 'train_dataset.pkl')):
        with open(os.path.join(pocket_dir, 'train_dataset.pkl'), 'rb') as f:
            train_dataset = pickle.load(f)
        with open(os.path.join(pocket_dir, 'test_dataset.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)

    else:
        train_dataset, test_dataset = create_dataset(pos_list, neg_list, pocket_dir, seq_cluster_map, fold_nr=0, type='seq')
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
    s=1024*1024
    # model.train()
    # epoch_train_losses = []
    # for epoch in range(epochs):
    #     for batch_idx, ((graph1, graph2), label) in enumerate(train_dataloader):
    #         if batch_idx==100:
    #             break
    #         torch.cuda.reset_peak_memory_stats(device)    
    #         print((torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()), torch.cuda.max_memory_allocated()/s)
    #         graph1, graph2, label = graph1.to(device), graph2.to(device), label.to(device)
            
    #         batch_indx1 = get_batch_idx(graph1).to(device)
    #         batch_indx2 = get_batch_idx(graph2).to(device)
    #         optimizer.zero_grad()
            
    #         output1 = model(graph1, batch_indx1)
    #         output2 = model(graph2, batch_indx2)
        
    #         loss, pos_dist, neg_dist = con_loss(output1, output2, label, loss_margin)

    #         print(loss.item())

    #         loss.backward()
    #         optimizer.step()
    #         continue

    #     break


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
        losses, pos_dists, neg_dists = [],[],[]

        progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Test Epoch: {epoch+1}")

        with torch.no_grad():
            for batch_idx, ((graph1, graph2), label) in progress_bar:
                # torch.cuda.reset_peak_memory_stats(device)  
                graph1, graph2, label = graph1.to(device), graph2.to(device), label.to(device)

                batch_indx1 = get_batch_idx(graph1).to(device)
                batch_indx2 = get_batch_idx(graph2).to(device)
                
                output1 = model(graph1, batch_indx1)
                output2 = model(graph2, batch_indx2)
                
                loss, pos_dist, neg_dist = con_loss(output1, output2, label, loss_margin)

                losses.append(loss.item())
                pos_dists.extend(pos_dist.cpu().numpy().tolist())
                neg_dists.extend(neg_dist.cpu().numpy().tolist()) 

        return {'loss' : np.mean(losses), 'pos_dist' : np.mean(pos_dists), 'neg_dist' : np.mean(neg_dists)}

    #epoch loop
    epoch_train_losses = []
    epoch_test_losses = []

    for epoch in range(epochs):  
        print("starting training runs")
        train_metrics = train(epoch)
        epoch_train_losses.append(train_metrics['loss'])
        wandb.log({'train_loss': train_metrics['loss'], 
                   'train_pos_dist': train_metrics['pos_dist'], 
                   'train_neg_dist': train_metrics['neg_dist'],
                   'epoch': epoch+1})
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_metrics['loss']:.4f}")

        if (epoch+1)%2==0:
            test_metrics = test(epoch)
            epoch_test_losses.append(test_metrics['loss'])
            wandb.log({'test_loss': test_metrics['loss'], 
                        'test_pos_dist': test_metrics['pos_dist'], 
                        'test_neg_dist': test_metrics['neg_dist'],
                        'epoch': epoch+1})
            print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_metrics['loss']:.4f}")

        scheduler.step() 

        current_lr = scheduler.get_last_lr()
        print(f"Current Learning Rate: {current_lr}")
  
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
            os.path.join(result_dir, 'model_3.pth.tar'))
        print(f"Model saved at epoch {epoch+1}")

if __name__=='__main__':
    main()

