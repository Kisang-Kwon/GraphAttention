import numpy as np

from collections import defaultdict
from utils.distance import euc_dist

class Atom:
    def __init__(self, atom_index, atom_type, coords, element):
        self.index = atom_index
        self.type = atom_type
        self.coords = coords # [x, y, z]
        self.element = element

    def get_coords(self):
        return self.coords

    def get_type(self):
        return self.type
    
    def get_element(self):
        return self.element


class Residue:
    ABBREVs = {
        'ARG': 'R', 'HIS': 'H', 'LYS': 'K', 'ASP': 'D', 'GLU': 'E', 
        'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'CYS': 'C',
        'GLY': 'G', 'PRO': 'P', 'ALA': 'A', 'ILE': 'I', 'LEU': 'L',
        'MET': 'M', 'PHE': 'F', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'SEC': 'U'
    }

    def __init__(self, name, number, chain):
        self.name = name
        self.number = number
        self.chain = chain
        self.atoms = []
        self.atom_cnt = {
            'C': 0,
            'N': 0, 
            'O': 0, 
            'S': 0,
            'Others': 0
        }
        self.c_alpha = None
        self.c_beta = None
        self.dist_ca = round(10e+10, 4)
        self.dist_cb = round(10e+10, 4)

    def add_atom(self, atom):
        """
        1. Atom instance를 residue의 atom list에 추가
        2. Atom의 type이 C alpha 또는 C beta일 경우, C alpha, C beta atom 별도 저장
        3. Residue에 포함되는 each atom의 개수 기록
            - 저장되는 Atom의 개수가 일반적인 Residue가 보유한 개수를 넘을 경우, 
              이후 featurize 단계에서 others index에 해당하는 숫자가 저장됨
              ex) C의 경우, atom 개수가 11개를 넘어가면 최종적으로 12라는 value가 저장됨
        """
        if atom.get_type() == 'CA':
            self.c_alpha = atom
        elif atom.get_type() == 'CB':
            self.c_beta = atom
        
        self.atoms.append(atom) # add Atom instance
        
        if atom.get_element() == 'C':
            if 11 < self.atom_cnt['C']:
                self.atom_cnt['C'] = 12
            self.atom_cnt['C'] += 1
        
        elif atom.get_element() == 'N':
            if 4 < self.atom_cnt['N']:
                self.atom_cnt['N'] = 5
            self.atom_cnt['N'] += 1
        
        elif atom.get_element() == 'O':
            if 4 < self.atom_cnt['O']:
                self.atom_cnt['O'] = 5
            self.atom_cnt['O'] += 1

        elif atom.get_element() == 'S':
            if 1 < self.atom_cnt['S']:
                self.atom_cnt['S'] = 2
            self.atom_cnt['S'] += 1
        else:
            if 2 < self.atom_cnt['Others']:
                self.atom_cnt['Others'] = 3
            self.atom_cnt['Others'] += 1

    def calc_centroid_distance(self, lig_centroid):
        self.dist_ca = euc_dist(lig_centroid, self.c_alpha.coords)
        if self.c_beta:
            self.dist_cb = euc_dist(lig_centroid, self.c_beta.coords)


class Ligand:
    def __init__(self, lig_name, chain, number):
        self.name = lig_name
        self.chain = chain
        self.number = number
        self.atoms = []
        self.connection = defaultdict(list)

    def add_atom(self, atom):
        self.atoms.append(atom)
    
    def get_centroid(self):
        coords = []
        for atom in self.atoms:
             coords.append(atom.coords)

        centroid = np.mean(coords, axis=0)
        
        return centroid