import argparse
import pickle
import shutil
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import dgl
from dgl.dataloading import GraphDataLoader
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb

from dataloader import get_dataloader, create_dataset
from graphpocket import GraphPocket
from model import ReceptorEncoderGVP, con_loss

sys.path.append('~/graphpocket/gnn')

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('../config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #set random seed manually
    torch.manual_seed(42)

    #create train and test datasets [log the paths in config] -- make all from config?
    pocket_dir = config['directories']['pockets']
    pos_list = config['directories']['pos_list']
    neg_list = config['directories']['neg_list']
    seq_cluster_map = config['directories']['cluster_map']
    result_dir = config['directories']['result_dir']

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
            return ReduceLROnPlateau(optimizer, mode=params['mode'], factor=params['factor'], patience=params['patience'], verbose=params['verbose'])
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
        
    loss_margin = config['loss']['margin']

    model_params = config['model']
    model = ReceptorEncoderGVP(
        in_scalar_size=model_params['input_scalar_size'],
        out_scalar_size=model_params['output_scalar_size'],
        vector_size=model_params['vector_size'],
        n_convs=model_params['n_convs'],
        dropout=model_params['dropout'],
    )

    opt_config = config['train']['optimizer']
    optimizer = get_optimizer(model.parameters(), opt_config)

    sched_config = config['train']['scheduler']
    scheduler = get_scheduler(optimizer, sched_config)

    train_dataset, test_dataset = create_dataset(pos_list, neg_list, pocket_dir, seq_cluster_map, fold_nr=0, type='seq')
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)

    #resume else create model, opt, and sched from config - wandb set up

    #train func 

    def train(model, dataloader, optimizer, device, margin):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        
        for ((graph1, graph2), label) in dataloader:
            graph1, graph2, label = graph1.to(device), graph2.to(device), label.to(device)
            
            output1 = model(graph1)
            output2 = model(graph2)
            
            loss, accuracy = con_loss(output1, output2, label, margin)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += accuracy
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        return avg_loss, avg_accuracy

    #test func



    #epoch loop

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(epochs):
        avg_loss, avg_accuracy = train(model, train_dataloader, optimizer, device, margin=1.0)
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(avg_accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Accuracy: {avg_accuracy}")

        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            torch.save(model.state_dict(), f"{result_dir}/model_epoch_{epoch+1}.pt")
        
        scheduler.step(avg_loss)  

    #maybe a resume function to start where stopped

    return


if __name__=='__main__':
    main()

