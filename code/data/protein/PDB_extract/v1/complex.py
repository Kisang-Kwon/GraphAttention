import os
import numpy as np
import csv

from collections import defaultdict

from .molecule import Atom, Residue, Ligand
from utils.dircheck import dircheck
from utils.distance import euc_dist


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
            
            if chain in Complex.USING_CHAIN_LIST[pid.upper()]:
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
        for lig_id in self.ligands:
            ligand = self.ligands[lig_id]
            lig_centroid = ligand.get_centroid()
            for l_atom in ligand.atoms:
                for res_id in self.residues:
                    residue = self.residues[res_id]
                    for r_atom in residue.atoms:
                        dist = euc_dist(l_atom.coords, r_atom.coords)
                        if dist <= 6.5:
                            residue.calc_centroid_distance(lig_centroid)
                            if res_id in self.pockets[lig_id]:
                                pass
                            else:
                                self.pockets[lig_id].append(res_id)
                            break
            
            self.make_pocket_files(Complex.POCKET_DIR, lig_id)
            M_feature, M_adjacency, mask = self.calc_features(lig_id)

            feat_path = os.path.join(Complex.FEATURE_DIR, f'{self.pdb}_{lig_id}')
            np.save(f'{feat_path}.npy', np.array([M_feature, M_adjacency, mask, self.pdb], dtype=object))

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
        M_feature = np.zeros(
            [Complex.max_poc_node, 34], dtype='float32'
        )
        M_adjacency = np.zeros(
            [Complex.max_poc_node, Complex.max_poc_node], dtype='float32'
        )
        mask = np.zeros([Complex.max_poc_node])
        
        for i, res_id in enumerate(self.pockets[lig_id]):
            """
            Atom별로 Others 인덱스에 들어가게 되는 숫자의 count가 되면 이를 처리해주기 위한 코드 구현
            """
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
            
            for j in range(i, len(self.pockets[lig_id])):
                res_id = self.pockets[lig_id][j]
                residue2 = self.residues[res_id]
                
                C_alpha_1 = residue.c_alpha.coords
                C_alpha_2 = residue2.c_alpha.coords

                dist = euc_dist(C_alpha_1, C_alpha_2)
                dist_weight = np.round(np.exp(-0.3*dist), 4)
                M_adjacency[i][j] = dist_weight
                M_adjacency[j][i] = dist_weight
                        
            mask[i] = 1
        
        return M_feature, M_adjacency, mask