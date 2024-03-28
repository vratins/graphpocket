import prody
import numpy as np
import torch
import dgl
from dgl.dataloading import GraphDataLoader

import pickle
import random
from sklearn.model_selection import KFold, GroupShuffleSplit

from graphpocket import GraphPocket


class GraphTupleDataset(dgl.data.DGLDataset):

    def __init__(self, name, pocket_list, pos_list, neg_list):
        self.pocket_list = pocket_list

        self.pos_list = list(filter(lambda p: p[0] in self.pocket_list and p[1] in self.pocket_list, pos_list))
        self.neg_list = list(filter(lambda p: p[0] in self.pocket_list and p[1] in self.pocket_list, neg_list))

        self.graphs = []
        self.labels = []

        #filter pos and neg based on pocket_list

        self.pocket_index_map = {}

        super().__init__(name=name)

    def __getitem__(self, index):

        pair, label = self.labels[index]
        pocket1 = self.graphs[self.pocket_index_map[pair[0]]]
        pocket2 = self.graphs[self.pocket_index_map[pair[1]]]

        return (pocket1, pocket2, label)

    def __len__(self):

        return len(self.labels)
    
    def process(self):

        pocket_to_graph = GraphPocket()

        for i, pocket in enumerate(self.pocket_list):
            graph = pocket_to_graph(pocket_path=f'../../dataset_graph/data/{pocket}/{pocket}_pocket.pdb')
            self.graphs.append(graph)
            self.pocket_index_map[pocket] = i

        for pos_pair in self.pos_list:
            self.labels.append((pos_pair, 1))
        for neg_pair in self.neg_list:
            self.labels.append((neg_pair, 0))

        random.shuffle(self.labels)

#function for dataloading tuples of the pockets from pocket lists - used to get dataloader from a dataset class

def create_dataset(pos_path, neg_path, fold_nr, type, n_folds=5, seed=42):
    
    #load in the list of pocket and corresponding sequence clusters
    with open('../cluster_map.pkl', 'rb') as file:
        pocket_seq = pickle.load(file)

    pocket_list = [pdb[0] + pdb[1] for pdb in list(pocket_seq.keys())]

    pockets = list(pocket_seq.keys())
    clusters = list(pocket_seq.values())

    if type == 'seq':
        split = GroupShuffleSplit(n_splits=n_folds, test_size=1.0/n_folds, random_state=seed)
        folds = list(split.split(pockets, groups=clusters))
        train_index, test_index = folds[fold_nr] #fold number?
        pocket_train, pocket_test = [pocket_list[i] for i in train_index], [pocket_list[i] for i in test_index]
    
    if type == 'random':
        split = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(split.split(pocket_list))
        train_index, test_index = folds[fold_nr]
        pocket_train, pocket_test = [pocket_list[i] for i in train_index], [pocket_list[i] for i in test_index]


    with open(pos_path) as f:
        pos_pairs = [line.split()[:2] for line in f.readlines()]
    with open(neg_path) as f:
        neg_pairs = [line.split()[:2] for line in f.readlines()]

    train_dataset = GraphTupleDataset(name='train', pocket_list=pocket_train, pos_list=pos_pairs, neg_list=neg_pairs) 
    test_dataset = GraphTupleDataset(name='test', pocket_list=pocket_test, pos_list=pos_pairs, neg_list=neg_pairs)

    return train_dataset, test_dataset

def collate_fn(samples):
    graphs1, graphs2, labels = map(list, zip(*samples))
    
    batched_graph1 = dgl.batch(graphs1)
    batched_graph2 = dgl.batch(graphs2)
    labels = torch.tensor(labels)
    
    return (batched_graph1, batched_graph2), labels

def get_dataloader(dataset, batch_size, num_workers, **kwargs):

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, **kwargs)

    return dataloader

