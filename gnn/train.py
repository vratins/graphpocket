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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_scatter import scatter_mean

import wandb

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
        if scheduler_config['type'] == "ReduceLROnPlateau":
            params = scheduler_config['params']
            return ReduceLROnPlateau(optimizer, mode=params['mode'], factor=params['factor'], patience=params['patience'])
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
    train_dataset, test_dataset = create_dataset(pos_list, neg_list, pocket_dir, seq_cluster_map, fold_nr=0, type='seq')
    print("Creating dataloaders..")
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)

    print("Size of train dataset: ", len(train_dataloader)*batch_size, "pairs")
    print("Size of test dataset: ", len(test_dataloader)*batch_size, "pairs") 

    #train func 
    def train(model, epoch, dataloader, optimizer, device, margin):
        model.train()
        losses, pos_dists, neg_dists = [], [], []

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch}')
        
        for batch_idx, ((graph1, graph2), label) in progress_bar:
            graph1, graph2, label = graph1.to(device), graph2.to(device), label.to(device)

            batch_indx1 = get_batch_idx(graph1).to(device)
            batch_indx2 = get_batch_idx(graph2).to(device)
            
            output1 = model(graph1, batch_indx1)
            output2 = model(graph2, batch_indx2)
            
            loss, pos_dist, neg_dist = con_loss(output1, output2, label, margin)

            losses.append(loss)
            pos_dists.append(pos_dist)
            neg_dists.append(neg_dist)    

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                            
        return {'loss' : torch.mean(losses).cpu().numpy(), 'pos_dist' : torch.mean(pos_dist).cpu().numpy(), 
                'neg_dist' : torch.mean(neg_dist).cpu().numpy()}

    #test func
    def test(model, epoch, dataloader, device, margin):
        model.eval()        
        losses, pos_dists, neg_dists = []

        progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f'Epoch {epoch}')

        with torch.no_grad():
            for batch_idx, ((graph1, graph2), label) in progress_bar:
                graph1, graph2, label = graph1.to(device), graph2.to(device), label.to(device)

                batch_indx1 = get_batch_idx(graph1).to(device)
                batch_indx2 = get_batch_idx(graph2).to(device)
                
                output1 = model(graph1)
                output2 = model(graph2)
                
                loss, pos_dist, neg_dist = con_loss(output1, output2, label, margin)

                losses.append(loss)
                pos_dists.append(pos_dist)
                neg_dists.append(neg_dist)

        return {'loss' : torch.mean(losses).cpu().numpy(), 'pos_dist' : torch.mean(pos_dist).cpu().numpy(), 
                'neg_dist' : torch.mean(neg_dist).cpu().numpy()}

    #epoch loop
    epoch_train_losses = []
    epoch_test_losses = []

    for epoch in range(epochs):  
        print("starting training runs")
        train_metrics = train(model, epoch, train_dataloader, optimizer, device, loss_margin)
        epoch_train_losses.append(train_metrics['loss'])
        wandb.log({'train_loss': train_metrics['loss'], 
                   'train_pos_dist': train_metrics['pos_dist'], 
                   'train_neg_dist': train_metrics['neg_dist'],
                   'epoch': epoch})
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_metrics['loss']:.4f}")

        test_metrics = test(model, epoch, test_dataloader, device, loss_margin)
        epoch_test_losses.append(test_metrics['loss'])
        wandb.log({'test_loss': test_metrics['loss'], 
                   'test_pos_dist': test_metrics['pos_dist'], 
                   'test_neg_dist': test_metrics['neg_dist'],
                   'epoch': epoch})
        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_metrics['loss']:.4f}")

        current_lr = optimizer.get_last_lr()
        print(f"Epoch {epoch+1}, Current Learning Rate(s): {current_lr}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
            os.path.join(result_dir, 'model.pth.tar'))
        print(f"Model saved at epoch {epoch+1}")

        scheduler.step(test_metrics['loss'])        


if __name__=='__main__':
    main()

