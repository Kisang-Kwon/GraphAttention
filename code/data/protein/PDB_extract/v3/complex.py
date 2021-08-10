import os
import sys
import numpy as np
import csv

from collections import defaultdict

from .molecule import Atom, Residue, Ligand
from utils.dircheck import dircheck
from utils.distance import euc_dist


KIDERA_FACTORS = {
    'ALA': [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.2, -0.08, 0.21, -0.48],
    'CYS': [0.12, -0.89, 0.45, -1.05, -0.71, 2.41, 1.52, -0.69, 1.13, 1.1],
    'GLU': [-1.45, 0.19, -1.61, 1.17, -1.31, 0.4, 0.04, 0.38, -0.35, -0.12],
    'ASP': [0.58, -0.22, -1.58, 0.81, -0.92, 0.15, -1.52, 0.47, 0.76, 0.7],
    'GLY': [1.46, -1.96, -0.23, -0.16, 0.1, -0.11, 1.32, 2.36, -1.66, 0.46],
    'PHE': [-0.21, 0.98, -0.36, -1.43, 0.22, -0.81, 0.67, 1.1, 1.71, -0.44],
    'ILE': [-0.73, -0.16, 1.79, -0.77, -0.54, 0.03, -0.83, 0.51, 0.66, -1.78],
    'HIS': [-0.41, 0.52, -0.28, 0.28, 1.61, 1.01, -1.85, 0.47, 1.13, 1.63],
    'LYS': [-0.34, 0.82, -0.23, 1.7, 1.54, -1.62, 1.15, -0.08, -0.48, 0.6],
    'MET': [-1.4, 0.18, -0.42, -0.73, 2.0, 1.52, 0.26, 0.11, -1.27, 0.27],
    'LEU': [-1.04, 0.0, -0.24, -1.1, -0.55, -2.05, 0.96, -0.76, 0.45, 0.93],
    'ASN': [1.14, -0.07, -0.12, 0.81, 0.18, 0.37, -0.09, 1.23, 1.1, -1.73],
    'GLN': [-0.47, 0.24, 0.07, 1.1, 1.1, 0.59, 0.84, -0.71, -0.03, -2.33],
    'PRO': [2.06, -0.33, -1.15, -0.75, 0.88, -0.45, 0.3, -2.3, 0.74, -0.28],
    'SER': [0.81, -1.08, 0.16, 0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],
    'ARG': [0.22, 1.27, 1.37, 1.87, -1.7, 0.46, 0.92, -0.39, 0.23, 0.93],
    'THR': [0.26, -0.7, 1.21, 0.63, -0.1, 0.21, 0.24, -1.15, -0.56, 0.19],
    'TRP': [0.3, 2.1, -0.72, -1.57, -1.16, 0.57, -0.48, -0.4, -2.3, -0.6],
    'VAL': [-0.74, -0.71, 2.04, -0.4, 0.5, -0.81, -1.07, 0.06, -0.46, 0.65],
    'TYR': [1.38, 1.48, 0.8, -0.56, -0.0, -0.68, -0.31, 1.03, -0.05, 0.53],
    'SEC': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
}

def calc_percentile(c_alpha_dists):
    p80 = np.percentile(c_alpha_dists, 80)
    p60 = np.percentile(c_alpha_dists, 60)
    p40 = np.percentile(c_alpha_dists, 40)
    p20 = np.percentile(c_alpha_dists, 20)

    return [p80, p60, p40, p20]


class Complex:
    USING_CHAIN_LIST = defaultdict(set)
    POCKET_DIR = None
    FEATURE_DIR = None
    max_poc_node = 100

    def __init__(self, pdb_id):
        self.pdb = pdb_id
        self.residues = dict()
        self.pockets = defaultdict(list)
        self.ligands = dict()
        self.filesaver = defaultdict(dict)
        self.c_alpha_d_percentile = defaultdict(list)
    
    @classmethod
    def set_using_chain_list(cls, f_chain_list):
        Fopen = open(f_chain_list)
        cread = csv.reader(Fopen)
        next(cread)
        for line in cread:
            pdb = line[0]
            chain = line[1]
            cls.USING_CHAIN_LIST[pdb].add(chain)
        Fopen.close()

    def parse_pdb(self, f_pdb):
        Fopen = open(f_pdb)
        
        for line in Fopen:
            if line.startswith('ATOM'):
                atom_index = int(line[6:11].lstrip())
                self.filesaver['ATOM'][atom_index] = line
            elif line.startswith('HETATM'):
                atom_index = int(line[6:11].lstrip())
                self.filesaver['HETATM'][atom_index] = line
        
        Fopen.close()

    def add_protein(self, protein):
        self.protein = protein

    def add_ligand(self, ligand):
        self.ligands.append(ligand)
    
    def set_protein_residues(self, pid):
        c_residue = 'None'
        for line in self.filesaver['ATOM'].values():
            atom_index = int(line[6:11].lstrip())
            atom_type = line[12:16].rstrip().lstrip()
            altLoc = line[16]
            res_name = line[17:20].rstrip().lstrip()
            chain = line[21]
            res_num = int(line[22:26].lstrip())
            x = float(line[30:38].lstrip())
            y = float(line[38:46].lstrip())
            z = float(line[46:54].lstrip())
            element = line[76:78].lstrip().rstrip()
            
            #if chain in Complex.USING_CHAIN_LIST[pid.upper()]:
            if altLoc == 'A' or altLoc == ' ':
                if c_residue == 'None':
                    c_residue = f'{res_name}_{res_num}_{chain}'
                    residue = Residue(res_name, res_num, chain)
                    atom = Atom(atom_index, atom_type, [x, y, z], element)
                    residue.add_atom(atom)

                elif c_residue == f'{res_name}_{res_num}_{chain}':
                    atom = Atom(atom_index, atom_type, [x, y, z], element)
                    residue.add_atom(atom)

                else:
                    if residue.c_alpha is not None:
                        self.residues[c_residue] = residue
                    c_residue = f'{res_name}_{res_num}_{chain}'
                    residue = Residue(res_name, res_num, chain)
                    atom = Atom(atom_index, atom_type, [x, y, z], element)
                    residue.add_atom(atom)
            else:
                pass
        
        if c_residue == 'None':
            return False
        else:
            if residue.c_alpha is not None:
                self.residues[c_residue] = residue
            return True

    def set_ligands(self, cid):
        """
        Save ligands
        """
        for line in self.filesaver['HETATM'].values():
            atom_index = int(line[6:11].lstrip())
            atom_type = line[12:16].rstrip().lstrip()
            altLoc = line[16]
            lig_name = line[17:20].rstrip().lstrip()
            lig_chain = line[21]
            lig_num = int(line[22:26].lstrip())
            x = float(line[30:38].lstrip())
            y = float(line[38:46].lstrip())
            z = float(line[46:54].lstrip())
            element = line[76:78].lstrip().rstrip()

            if lig_name == cid :
                if altLoc == 'A' or altLoc == ' ':
                    lig_id = f'{lig_name}_{lig_num}_{lig_chain}'
                    if lig_id in self.ligands.keys():
                        atom = Atom(atom_index, atom_type, [x, y, z], element)
                        self.ligands[lig_id].add_atom(atom)
                    else:
                        self.ligands[lig_id] = Ligand(
                            lig_name, lig_chain, lig_num
                        )

    def find_pocket(self):
        """
        Pocket Residue의 정의: 
            Ligand Centroid로부터 Residue C alpha 사이의 거리가 12 Angstrom 이하인 Residue

        C alpha distance percentile 계산 -> Protein feature에 반영
        """
        for lig_id in self.ligands:
            try:
                ligand = self.ligands[lig_id]
                lig_centroid = ligand.get_centroid()
                c_alpha_dists = list()
                for res_id in self.residues:
                    residue = self.residues[res_id]
                    residue.calc_centroid_distance(lig_centroid)
                    if residue.dist_ca <= 12:
                        self.pockets[lig_id].append(res_id)
                        c_alpha_dists.append(residue.dist_ca)
                
                self.c_alpha_d_percentile[lig_id] = calc_percentile(c_alpha_dists)
                
                self.make_pocket_files(Complex.POCKET_DIR, lig_id)
                M_feat, M_adj, mask, dist_coeff, res_names = self.calc_features(lig_id)
                feat_path = os.path.join(Complex.FEATURE_DIR, f'{self.pdb}_{lig_id}')

                np.save(
                    f'{feat_path}.npy',
                    np.array(
                        [M_feat, M_adj, mask, dist_coeff, self.pdb, res_names],
                        dtype=object
                    )
                )
            except:
                print(self.pdb ,lig_id)
                sys.stderr.write(f'[{self.pdb},{lig_id}] Error in find_pocket process... \n')

    def make_pocket_files(self, pocdir, lig_id):
        dircheck(os.path.join(pocdir, self.pdb))
        poc_path = os.path.join(pocdir, self.pdb, f'{lig_id}.pdb')
        POC = open(f'{poc_path}', 'w')
        
        for line in self.filesaver['ATOM'].values():
            altLoc = line[16]
            res_name = line[17:20].rstrip().lstrip()
            res_chain = line[21]
            res_num = int(line[22:26].lstrip())
            if altLoc == 'A' or altLoc == ' ':
                res_id = f'{res_name}_{res_num}_{res_chain}'
                if res_id in self.pockets[lig_id]:
                    POC.write(line)
            
        POC.close()
        
    def calc_features(self, lig_id):
        """
            Feature:
                Each Residue 별 Atom Count
                C alpha & beta의 Relative Location
                Kidera Factor

            Atom별로 Others 인덱스에 들어가게 되는 숫자의 count가 되면 이를 처리해주기 위한 코드 구현
        """
        M_feature = np.zeros(
            [Complex.max_poc_node, 44], dtype='float32'
        )
        M_adjacency = np.zeros(
            [Complex.max_poc_node, Complex.max_poc_node], dtype='float32'
        )
        dist_coeff = np.zeros([Complex.max_poc_node], dtype='float32')
        mask = np.zeros([Complex.max_poc_node])
        residue_names = []
        
        for i, res_id in enumerate(self.pockets[lig_id]):
            residue = self.residues[res_id]
            C_idx = int(residue.atom_cnt['C'])       # Possible number of C atoms [0,1,2,3,4,5,6,7,8,9,10,11,others]
            N_idx = int(residue.atom_cnt['N']) + 13  # Possible number of N atoms [0,1,2,3,4,others]
            O_idx = int(residue.atom_cnt['O']) + 19  # Possible number of O atoms [0,1,2,3,4,others]
            S_idx = int(residue.atom_cnt['S']) + 25  # Possible number of S atoms [0,1,others]
            OTHERS_idx = int(residue.atom_cnt['Others']) + 28  # Possible number of OTHER atoms [0,1,2,others] // ~ index 31

            M_feature[i][C_idx] = 1
            M_feature[i][N_idx] = 1
            M_feature[i][O_idx] = 1
            M_feature[i][S_idx] = 1
            M_feature[i][OTHERS_idx] = 1
            
            if residue.name == 'GLY':
                M_feature[i][32] = 1
                M_feature[i][33] = 0
            else:
                if residue.dist_ca <= residue.dist_cb:
                    M_feature[i][32] = 1
                    M_feature[i][33] = 0
                else:
                    M_feature[i][32] = 0
                    M_feature[i][33] = 1
            
            M_feature[i][-10:] = KIDERA_FACTORS[residue.name]

            for j in range(i, len(self.pockets[lig_id])):
                res_id = self.pockets[lig_id][j]
                residue2 = self.residues[res_id]
                
                C_alpha_1 = residue.c_alpha.coords
                C_alpha_2 = residue2.c_alpha.coords

                if i == j:
                    M_adjacency[i][j] = 1
                else:
                    dist = euc_dist(C_alpha_1, C_alpha_2)
                    if dist <= 7:
                        M_adjacency[i][j] = 1
                        M_adjacency[j][i] = 1
            
            mask[i] = 1

            const = 2
            if self.c_alpha_d_percentile[lig_id][0] <= residue.dist_ca:
                dist_coeff[i] = const * 0.2
            elif self.c_alpha_d_percentile[lig_id][1] <= residue.dist_ca:
                dist_coeff[i] = const * 0.4
            elif self.c_alpha_d_percentile[lig_id][2] <= residue.dist_ca:
                dist_coeff[i] = const * 0.6
            elif self.c_alpha_d_percentile[lig_id][3] <= residue.dist_ca:
                dist_coeff[i] = const * 0.8
            else:
                dist_coeff[i] = const * 1.0
            
            residue_names.append(f'{residue.name}_{residue.chain}_{residue.number}')

        return M_feature, M_adjacency, mask, dist_coeff, residue_names