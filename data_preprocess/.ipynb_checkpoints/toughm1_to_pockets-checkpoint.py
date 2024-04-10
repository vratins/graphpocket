"""
Script to process atom-based pockets obtained from fpocket and convert to residue-based pockets
"""

import warnings
import os
import shutil
from tqdm import tqdm

from Bio import BiopythonWarning
from Bio.PDB import PDBParser, PDBIO, NeighborSearch, Select

warnings.simplefilter('ignore', BiopythonWarning)

class ResidueSelect(Select):
    #class to return biopython object of selected residue
    def __init__(self, residues):
        self.residues = residues

    def accept_residue(self, residue):
        return residue in self.residues
    
def find_residue(protein_path, pocket_path, output_path):
    parser = PDBParser()

    #read in protein and fpocket generated atom-based pocket files
    protein_structure = parser.get_structure("Protein", protein_path)
    pocket_structure = parser.get_structure("Pocket", pocket_path)

    protein_atoms = [atom for atom in protein_structure.get_atoms()]
    pocket_atoms = [atom for atom in pocket_structure.get_atoms()]

    #find all residues near the pocket
    searcher = NeighborSearch(protein_atoms)
    intersecting_residues = set()

    #threshold of 0.1-0.3 A should give us the exact residues of the pocket
    for pocket_atom in pocket_atoms:
        close_atoms = searcher.search(pocket_atom.coord, 1.0, 'A')
        for atom in close_atoms:
            intersecting_residues.add(atom.get_parent())

    io = PDBIO()
    io.set_structure(protein_structure)
    io.save(f"{output_path}.pdb", ResidueSelect(intersecting_residues))   

def check_and_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def copy_file(source, destination):
    shutil.copy(source, destination)

def process_pockets(pocket_list, input_dir, output_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    check_and_create_dir(output_dir)

    #iterate of the list of pockets provided by tough-m1
    for pdb, number in tqdm(pocket_list):
        pdb_path = os.path.join(input_dir, str(pdb))
        pocket_path = os.path.join(pdb_path, f"{pdb}_out/pockets", f"pocket{number-1}_atm.pdb")
        pdb_file = os.path.join(pdb_path, f"{pdb}.pdb")
        
        if not os.path.exists(pocket_path) or not os.path.exists(pdb_path):
            print(f"Missing files for ID {pdb} and pocket number {number-1}. Skipping...")
            continue
        
        output_pdb_path = os.path.join(output_dir, str(pdb))
        check_and_create_dir(output_pdb_path) 
        
        copy_file(pdb_file, os.path.join(output_pdb_path, f"{pdb}.pdb"))
        
        output_pocket_path = os.path.join(output_pdb_path, f"{pdb}_pocket")
        
        find_residue(pdb_file, pocket_path, output_pocket_path)

pocket_list = []

#add path as argument later
with open('../../dataset_graph/TOUGH-M1/TOUGH-M1_pocket.list', 'r') as file:
    for line in file:
        parts = line.strip().split()
        pdbid = parts[0]
        pocket_number = int(parts[1])

        pocket_list.append((pdbid, pocket_number))

process_pockets(pocket_list, '../../dataset_graph/TOUGH-M1/TOUGH-M1_dataset', '../../dataset_graph/data')
