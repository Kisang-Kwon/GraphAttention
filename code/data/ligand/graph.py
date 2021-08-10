#!/usr/bin/env python
# coding: utf-8
'''
Last update: 21.06.21. by KS.Kwon
'''

import os
import sys
import time
import numpy as np

from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from collections import defaultdict
from .feature import get_atom_feature, get_bond_feature


class MolGraph:
    def __init__(self, cid, smiles, **kwargs):
        self.atom_feature_set = dict()
        self.bond_feature_set = dict()
        self.nodes = [] # dict of lists of nodes, keyed by node type (atom or bond)
        self.n_atom_feature = None
        self.n_bond_feature = None
        self.n_bond = None

        mol = Chem.MolFromSmiles(smiles)

        self.set_features()

        if mol is None:
            print(f'{cid}: Cannot generate MOL object')
            self.mol = None
        else:
            self.mol = True
            self.atom_features = get_atom_feature(
                mol=mol, **self.atom_feature_set
            )
            self.bond_features = get_bond_feature(
                mol=mol, **self.bond_feature_set
            )
            
            for atom in mol.GetAtoms():
                atom_neighbors = atom.GetNeighbors()
                bond_neighbors = atom.GetBonds()
                self.nodes.append(
                    Atom(atom_neighbors, bond_neighbors, atom.GetIdx())
                ) # The index of list the Atom / Bond instances are located in is same as rdkit index
            
            self.n_atom_feature = len(self.atom_features[-1])
            self.n_bond_feature = len(self.bond_features[-1])
            self.n_bond = len(mol.GetBonds())


    def set_features(self):
        self.flag = 2 + 4 + 8 + 64 + 8192 + 32768 + 65536
        
        # Atom feature
        if self.flag & 2: # 1
            self.atom_feature_set['atom_type'] = True
        if self.flag & 4: # 2 
            self.atom_feature_set['atom_degree'] = True
        if self.flag & 8: # 3 
            self.atom_feature_set['hybridization'] = True
        if self.flag & 16: # 4
            self.atom_feature_set['partial_charge'] = True
        if self.flag & 32: # 5
            self.atom_feature_set['chirality'] = True
        if self.flag & 64: # 6
            self.atom_feature_set['aromaticity'] = True
        if self.flag & 128: # 7
            self.atom_feature_set['atom_mass'] = True
        if self.flag & 256: # 8
            self.atom_feature_set['implicit_valence'] = True
        if self.flag & 512: # 9
            self.atom_feature_set['vdw'] = True
        if self.flag & 1024: # 10
            self.atom_feature_set['num_H'] = True
        if self.flag & 2048: # 11
            self.atom_feature_set['H_donor'] = True
        if self.flag & 4096: # 12
            self.atom_feature_set['H_acceptor'] = True
        
        # Bond feature
        if self.flag & 8192: # 13
            self.bond_feature_set['bond_type'] = True
        if self.flag & 16384: # 14
            self.bond_feature_set['stereotype'] = True
        if self.flag & 32768: # 15
            self.bond_feature_set['conjugate'] = True
        if self.flag & 65536: # 16
            self.bond_feature_set['inRing'] = True


    def make_matrix(self, max_mol_node, max_mol_degree, max_bond):
        M_atom_feat = np.zeros((max_mol_node, self.n_atom_feature))
        M_bond_feat = np.zeros((max_bond, self.n_bond_feature))
        M_atom_adj = np.zeros((max_mol_node, max_mol_node))
        M_bond_adj = np.zeros((max_mol_node, max_bond))
        M_mol_deg = np.zeros((max_mol_node, max_mol_degree))
        mask = np.zeros((max_mol_node,))

        for i in range(max_mol_node):
            
            if i < len(self.nodes):
                atom = self.nodes[i]
                #print(atom.GetBonds())
                M_atom_feat[i] = self.atom_features[i]
                
                M_atom_adj[i][i] = 1
                neighbors = atom.atom_neighbors
                for neighbor in neighbors:
                    M_atom_adj[i][neighbor.GetIdx()] = 1
                    
                bonds = atom.bond_neighbors
                for bond in bonds:
                    M_bond_adj[i][bond.GetIdx()] = 1
                
                mask[i] = 1

        for i in range(self.n_bond):
            M_bond_feat[i] = self.bond_features[i]

        return M_atom_feat, M_bond_feat, M_atom_adj, M_bond_adj, mask


class Atom:
    __slots__ = ['atom_neighbors', 'bond_neighbors', 'atom_idx']
    
    def __init__(self, atom_neighbors, bond_neighbors, atom_idx):
        self.atom_neighbors = atom_neighbors
        self.bond_neighbors = bond_neighbors
        self.atom_idx = atom_idx