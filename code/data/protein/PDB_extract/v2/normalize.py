#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import csv
import argparse


def dircheck(dirpath):
    dirpath = os.path.abspath(dirpath)
    dir_list = dirpath.split(os.path.sep)
    for i, val in enumerate(dir_list):
        if i == 0: continue
        dirpath_ = '/'.join(dir_list[:i+1])
        if os.path.isdir(dirpath_) is False:
            os.mkdir(dirpath_)


def get_pocket_features_stat(inputdir, mode):
    all_pocket_feature = []
    for npy in os.listdir(inputdir):
        if npy.endswith('.npy'):
            filepath = os.path.join(inputdir, npy)
            features, _, mask, _, _ = np.load(filepath, allow_pickle=True)
            residues = int(np.sum(mask))
            all_pocket_feature.extend(features[:residues])
            
    all_pocket_feature = np.array(all_pocket_feature, dtype=np.float32)

    if mode == 'MinMax':
        max_vec = np.max(all_pocket_feature, axis=0)
        min_vec = np.min(all_pocket_feature, axis=0)
        
        return max_vec, min_vec
    
    elif mode == 'Zscore':
        mean_vec = np.mean(all_pocket_feature, axis=0)
        var_vec = np.var(all_pocket_feature, axis=0)

        return mean_vec, var_vec


def normalization(input_fpath, output_fpath, mode=None, norm_p1=None, norm_p2=None):
    in_feature, M_adj, mask, cent_dist, pid = np.load(input_fpath, allow_pickle=True)
    
    out_feature = np.zeros(in_feature.shape)
    feature_dims = out_feature.shape[1]
    
    for i in range(int(np.sum(mask))):  # for real residue
        for j in range(feature_dims):   # for each feature
            
            if mode == 'MinMax':
                min_vec = norm_p1
                max_vec = norm_p2
                
                if max_vec[j] - min_vec[j] != 0:
                    out_feature[i][j] = min_max_norm(in_feature[i][j], min_vec[j], max_vec[j])
                elif max_vec[j] != 0:
                    out_feature[i][j] = float(in_feature[i][j]) / max_vec[j]
                else:
                    out_feature[i][j] = float(in_feature[i][j])
            
            elif mode == 'Zscore':
                mean_vec = norm_p1
                var_vec = norm_p2
                
                if var_vec[j] != 0:
                    out_feature[i][j] = z_score_norm(in_feature[i][j], mean_vec[j], var_vec[j])
                else:
                    out_feature[i][j] = float(in_feature[i][j])

    output = np.array([out_feature, M_adj, mask, cent_dist, pid])
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
        norm_p1, norm_p2 = np.load(args.minmax, allow_pickle=True)
    elif args.zscore:
        norm_p1, norm_p2 = np.load(args.zscore, allow_pickle=True)
    else:
        norm_p1, norm_p2 = get_pocket_features_stat(args.inputdir, args.mode)

        params_dir = '/'.join(os.path.abspath(args.outdir).split('/')[:-3])
        if args.mode == 'MinMax':
            O_params = os.path.join(params_dir, 'minmax.npy')
        elif args.mode == 'Zscore':
            O_params = os.path.join(params_dir, 'z_score.npy')

        np.save(O_params, np.array([norm_p1, norm_p2]))

    for npy in os.listdir(args.inputdir):
        if npy.endswith('.npy'):
            input_fpath = os.path.join(args.inputdir, npy)
            output_fpath = os.path.join(args.outdir, npy)
            normalization(input_fpath=input_fpath, output_fpath=output_fpath, mode=args.mode, norm_p1=norm_p1, norm_p2=norm_p2)