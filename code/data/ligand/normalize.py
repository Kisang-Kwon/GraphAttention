import os
import sys
import argparse

import numpy as np

from utils.dircheck import dircheck

def get_ligand_features_stat(ligand_dir, mode):
    all_atom_feature = []
    all_bond_feature = []
    for npy in os.listdir(ligand_dir):
        if npy.endswith('.npy'):
            filepath = os.path.join(ligand_dir, npy)
            atom_features, bond_features, _, _, atom_mask, _ = np.load(
                filepath, allow_pickle=True
            )
            atoms = int(np.sum(atom_mask))
            all_atom_feature.extend(atom_features[:atoms])

            for bond in bond_features:
                if np.max(bond) == 0 and np.min(bond) == 0:
                    break
                else:
                    all_bond_feature.append(bond)

    all_atom_feature = np.array(all_atom_feature, dtype=np.float32)
    all_bond_feature = np.array(all_bond_feature, dtype=np.float32)

    if mode == 'MinMax':
        atom_max = np.max(all_atom_feature, axis=0)
        atom_min = np.min(all_atom_feature, axis=0)
        bond_max = np.max(all_bond_feature, axis=0)
        bond_min = np.min(all_bond_feature, axis=0)
    
        return atom_max, atom_min, bond_max, bond_min

    elif mode == 'Zscore':
        atom_mean = np.mean(all_atom_feature, axis=0)
        atom_var = np.var(all_atom_feature, axis=0)
        bond_mean = np.mean(all_bond_feature, axis=0)
        bond_var = np.var(all_bond_feature, axis=0)

        return atom_mean, atom_var, bond_mean, bond_var


def normalization(input_fpath, output_fpath, mode=None, atom_norm_p1=None, atom_norm_p2=None, bond_norm_p1=None, bond_norm_p2=None):
    atom_feature, bond_feature, M_atom_adj, M_bond_adj, mask, cid = np.load(input_fpath, allow_pickle=True)
    
    out_atom_feature = np.zeros(atom_feature.shape)
    out_bond_feature = np.zeros(bond_feature.shape)
    
    atom_feature_dims = out_atom_feature.shape[1]
    n_bonds = out_bond_feature.shape[0]
    bond_feature_dims = out_bond_feature.shape[1]
    
    for i in range(int(np.sum(mask))):  # for real atoms
        for j in range(atom_feature_dims):   # for each feature
            
            if mode == 'MinMax':
                min_vec = atom_norm_p1
                max_vec = atom_norm_p2
                
                if max_vec[j] - min_vec[j] != 0:
                    out_atom_feature[i][j] = min_max_norm(atom_feature[i][j], min_vec[j], max_vec[j])
                elif max_vec[j] != 0:
                    out_atom_feature[i][j] = float(atom_feature[i][j]) / max_vec[j]
                else:
                    out_atom_feature[i][j] = float(atom_feature[i][j])
            
            elif mode == 'Zscore':
                mean_vec = atom_norm_p1
                var_vec = atom_norm_p2
                
                if var_vec[j] != 0:
                    out_atom_feature[i][j] = z_score_norm(atom_feature[i][j], mean_vec[j], var_vec[j])
                else:
                    out_atom_feature[i][j] = float(atom_feature[i][j])

    for i in range(n_bonds):
        if np.max(bond_feature[i]) == 0 and np.min(bond_feature[i]) == 0:
            break
        else:
            for j in range(bond_feature_dims):

                if mode == 'MinMax':
                    min_vec = bond_norm_p1
                    max_vec = bond_norm_p2
                    
                    if max_vec[j] - min_vec[j] != 0:
                        out_bond_feature[i][j] = min_max_norm(bond_feature[i][j], min_vec[j], max_vec[j])
                    elif max_vec[j] != 0:
                        out_bond_feature[i][j] = float(bond_feature[i][j]) / max_vec[j]
                    else:
                        out_bond_feature[i][j] = float(bond_feature[i][j])
                
                elif mode == 'Zscore':
                    mean_vec = bond_norm_p1
                    var_vec = bond_norm_p2

                    if var_vec[j] != 0:
                        out_bond_feature[i][j] = z_score_norm(bond_feature[i][j], mean_vec[j], var_vec[j])
                    else:
                        out_bond_feature[i][j] = float(bond_feature[i][j])

    output = np.array([out_atom_feature, out_bond_feature, M_atom_adj, M_bond_adj, mask, cid])
    np.save(output_fpath, output)


def min_max_norm(val, min_, max_):
    return (float(val) - float(min_)) / (float(max_) - float(min_))


def z_score_norm(val, mean, var):
    return (float(val) - float(mean)) / np.sqrt(float(var))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('-m', dest='mode', type=str, required=True)
    parser.add_argument('-i', dest='inputdir', type=str, required=True)
    parser.add_argument('-o', dest='outdir', type=str, required=True)
    parser.add_argument('--minmax', dest='minmax', type=str, default=False)
    parser.add_argument('--zscore', dest='zscore', type=str, default=False)
    args = parser.parse_args()

    print(args)

    if os.path.isdir(args.outdir) is False:
        dircheck(args.outdir)
    
    if args.minmax:
        atom_norm_p1, atom_norm_p2, bond_norm_p1, bond_norm_p2 = np.load(args.minmax, allow_pickle=True)
    elif args.zscore:
        atom_norm_p1, atom_norm_p2, bond_norm_p1, bond_norm_p2 = np.load(args.zscore, allow_pickle=True)
    else:
        atom_norm_p1, atom_norm_p2, bond_norm_p1, bond_norm_p2 = get_ligand_features_stat(args.inputdir, args.mode)

        params_dir = '/'.join(os.path.abspath(args.outdir).split('/')[:-3])
        if args.mode == 'MinMax':
            O_params = os.path.join(params_dir, 'minmax.npy')
        elif args.mode == 'Zscore':
            O_params = os.path.join(params_dir, 'z_score.npy')

        np.save(O_params, np.array([atom_norm_p1, atom_norm_p2, bond_norm_p1, bond_norm_p2]))

    for npy in os.listdir(args.inputdir):
        if npy.endswith('.npy'):
            input_fpath = os.path.join(args.inputdir, npy)
            output_fpath = os.path.join(args.outdir, npy)
            normalization(input_fpath=input_fpath, output_fpath=output_fpath, mode=args.mode, atom_norm_p1=atom_norm_p1, atom_norm_p2=atom_norm_p2, bond_norm_p1=bond_norm_p1, bond_norm_p2=bond_norm_p2)
