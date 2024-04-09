from pathlib import Path
import prody
import numpy as np
import torch
import dgl

prody.confProDy(verbosity='none')

from typing import Iterable, Union, List, Dict

class Unparsable(Exception):
    pass

#processing code adapted from Ian Dunn: https://github.com/Dunni3/keypoint-diffusion/blob/main/data_processing/pdbbind_processing.py

class GraphPocket:
#callable class to read a pocket and output the graph
    
    def __init__(self, k, algorithm):

        #hard code element map and k for graph
        self.rec_elements = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'other':4}
        self.threshold_k = k
        self.algorithm = algorithm

    def __call__(self, pocket_path):

        pocket = parse_pocket(pocket_path)
        positions, features, residues = get_pocket_atoms(pocket, self.rec_elements)

        graph = build_pocket_graph(positions, features, residues, self.threshold_k, self.algorithm)

        return graph
        

#function to parse the receptors
def parse_pocket(pocket_path): #reads in pdb file of a receptor(binding pocket) into a prody AtomGroup
    
    receptor = prody.parsePDB(str(pocket_path))

    if receptor is None: #errors in reading in a pocket
        raise Unparsable
    
    return receptor

#function to return atom positions, features
def get_pocket_atoms(rec_atoms, element_map):

    #position, features and indices for all pocket atoms
    rec_atom_positions = rec_atoms.getCoords()
    rec_res_indices = rec_atoms.getResindices()
    rec_atom_features, other_atoms_mask = receptor_featurizer(element_map=element_map, rec_atoms=rec_atoms)

    #convert positions and features to tensors
    rec_atom_positions = torch.tensor(rec_atom_positions).float()
    rec_atom_features = torch.tensor(rec_atom_features).float()
    rec_res_indices = torch.tensor(rec_res_indices).float()

    # remove "other" atoms from the receptor
    rec_atom_positions = rec_atom_positions[~other_atoms_mask]
    rec_atom_features = rec_atom_features[~other_atoms_mask]
    rec_res_indices = rec_res_indices[~other_atoms_mask]

    return rec_atom_positions, rec_atom_features, rec_res_indices


#function to featurize the receptor atoms
def receptor_featurizer(element_map, rec_atoms, protein_atom_elements = None):

    if rec_atoms is None and protein_atom_elements is None:
        raise ValueError
    
    if protein_atom_elements is None:
        protein_atom_elements: np.ndarray = rec_atoms.getElements()

    #one-hot encode atom elements
    onehot_elements = one_hot_encode(protein_atom_elements, element_map)

    #mask "other" atoms
    other_atoms_mask = torch.tensor(onehot_elements[:, -1] == 1).bool()

    #remove "other" category from onehot_elements, assuming they are last in the one-hot encoding
    protein_atom_features = onehot_elements[:, :-1]

    return protein_atom_features, other_atoms_mask


#function to one-hot encode all atoms of the receptor
def one_hot_encode(atom_elements: Iterable, element_map: Dict[str, int]):

    def element_to_idx(element_str, element_map=element_map):
        try:
            return element_map[element_str]
        except KeyError:
            return element_map['other']

    element_idxs = np.fromiter((element_to_idx(element) for element in atom_elements), int)
    onehot_elements = np.zeros((element_idxs.size, len(element_map.values())))
    onehot_elements[np.arange(element_idxs.size), element_idxs] = 1

    return onehot_elements

#function to build a graph from receptor atoms using dgl
def build_pocket_graph(atom_positions: torch.Tensor, atom_features: torch.Tensor, res_index: torch.Tensor, k: int, edge_algorithm: str):
    #add functionality for radius graphs too

    g = dgl.knn_graph(atom_positions, k=k, algorithm=edge_algorithm, dist='euclidean', exclude_self=True)
    g.ndata['x_0'] = atom_positions
    g.ndata['h_0'] = atom_features

    source_nodes, destination_nodes = g.edges()
    same_residue = res_index[source_nodes] == res_index[destination_nodes]
    edge_feature = same_residue.float()
    g.edata['a'] = edge_feature.view(-1,1)
    
    return g
