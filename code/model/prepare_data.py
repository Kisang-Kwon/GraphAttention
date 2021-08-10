import os
import csv
import numpy as np
from multiprocessing import Pool

from utils.get_time import get_time

def set_data_list(f_input_list, pocket_dir, ligand_dir):
    Fopen = open(f_input_list)
    csvreader = csv.reader(Fopen)

    input_list = []
    for line in csvreader:
        f_poc = os.path.join(pocket_dir, f'{line[0]}.npy')
        f_lig = os.path.join(ligand_dir, f'{line[1]}.npy')
        label = line[2]
        smiles = line[3]
        input_list.append([f_poc, f_lig, label, smiles])
    
    return input_list

def get_batch_data(dataset, batch_size):
    b_poc_feat = []
    b_poc_adj = []
    b_poc_d_score = []
    b_poc_mask = []
    b_pid = []
    b_res_names = []
    b_lig_atom_feat = []
    b_lig_atom_adj = []
    b_lig_bond_feat = []
    b_lig_bond_adj = []
    b_lig_mask = []
    b_label = []
    b_smiles = []
    b_cid = []
            
    for poc_path, lig_path, label, smiles in dataset:
        if os.path.isfile(poc_path):
            if os.path.isfile(lig_path):
                poc_data = np.load(poc_path, allow_pickle=True)
                lig_data = np.load(lig_path, allow_pickle=True)

                poc_feat = poc_data[0]
                poc_adj = poc_data[1]
                poc_mask = poc_data[2]
                poc_d_score = poc_data[3]
                pid = poc_data[4]
                res_names = poc_data[5]

                lig_atom_feat = lig_data[0]
                lig_bond_feat = lig_data[1]
                lig_atom_adj = lig_data[2]
                lig_bond_adj = lig_data[3]
                lig_mask = lig_data[4]
                cid = lig_data[5]

                if len(b_poc_feat) == batch_size:
                    b_poc_feat = np.array(b_poc_feat, dtype='float32')
                    b_poc_adj = np.array(b_poc_adj, dtype='float32')
                    b_poc_d_score = np.array(b_poc_d_score, dtype='float32')
                    b_poc_mask = np.array(b_poc_mask, dtype='float32')

                    b_lig_atom_feat = np.array(b_lig_atom_feat, dtype='float32')
                    b_lig_atom_adj = np.array(b_lig_atom_adj, dtype='float32')
                    b_lig_bond_feat = np.array(b_lig_bond_feat, dtype='float32')
                    b_lig_bond_adj = np.array(b_lig_bond_adj, dtype='float32')
                    b_lig_mask = np.array(b_lig_mask, dtype='float32')
                    b_label = np.array(b_label, dtype='float32')
                    
                    yield (
                        b_poc_feat, b_poc_adj, b_poc_mask, b_poc_d_score, b_pid, 
                        b_res_names, b_lig_atom_feat, b_lig_bond_feat, b_lig_atom_adj,
                        b_lig_bond_adj, b_lig_mask, b_label, b_smiles, b_cid
                    )

                    b_poc_feat = []
                    b_poc_adj = []
                    b_poc_mask = []
                    b_poc_d_score = []
                    b_pid = []
                    b_res_names = []
                    b_lig_atom_feat = []
                    b_lig_atom_adj = []
                    b_lig_bond_feat = []
                    b_lig_bond_adj = []
                    b_lig_mask = []
                    b_label = []
                    b_smiles = []
                    b_cid = []

                b_poc_feat.append(poc_feat)
                b_poc_adj.append(poc_adj)
                b_poc_mask.append(poc_mask)
                b_poc_d_score.append(poc_d_score)
                b_pid.append(pid)
                b_res_names.append(res_names)
                b_lig_atom_feat.append(lig_atom_feat)
                b_lig_atom_adj.append(lig_atom_adj)
                b_lig_bond_feat.append(lig_bond_feat)
                b_lig_bond_adj.append(lig_bond_adj)
                b_lig_mask.append(lig_mask)
                b_label.append(make_label(label))
                b_smiles.append(smiles)
                b_cid.append(cid)

    if 0 < len(b_poc_feat):
        b_poc_feat = np.array(b_poc_feat, dtype='float32')
        b_poc_adj = np.array(b_poc_adj, dtype='float32')
        b_poc_d_score = np.array(b_poc_d_score, dtype='float32')
        b_poc_mask = np.array(b_poc_mask, dtype='float32')
        b_lig_atom_feat = np.array(b_lig_atom_feat, dtype='float32')
        b_lig_atom_adj = np.array(b_lig_atom_adj, dtype='float32')
        b_lig_bond_feat = np.array(b_lig_bond_feat, dtype='float32')
        b_lig_bond_adj = np.array(b_lig_bond_adj, dtype='float32')
        b_lig_mask = np.array(b_lig_mask, dtype='float32')
        b_label = np.array(b_label, dtype='float32')
        
        yield (
            b_poc_feat, b_poc_adj, b_poc_mask, b_poc_d_score, b_pid, 
            b_res_names, b_lig_atom_feat, b_lig_bond_feat, b_lig_atom_adj,
            b_lig_bond_adj, b_lig_mask, b_label, b_smiles, b_cid
        )


def make_label(label):
    lab_arr = np.zeros([2])
    label = int(label)
    lab_arr[label] = 1
    
    return lab_arr