import argparse
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import dgl
import numpy as np
import torch
import yaml
from dgl.dataloading import GraphDataLoader

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

    #resume else create model, opt, and sched from config - wandb set up

    #train func - dont forget to edit model.py to return accuracy

    #test func

    #epoch loop

    #maybe a resume function to start where stopped

    return


if __name__=='__main__':
    main()

