#!/usr/bin/env python
# coding: utf-8
'''
Last update: 21.01.22. by KS.Kwon

[21.01.22.]
- Protein residue_names data added to protein batch_data

[21.01.14.]
- Model structure
    - Graph layer:          GAT
        - Protein layer:    2
        - Ligand layer:     2 
    - Interaction layer:    1
    - Output layer:         1
    - Dropout:              Yes
        - Keep probs:       0.5
    - FP generating method: GAT._convToFP
    - Loss function:        Cross-entropy (softmax)
    - L2 regularization:    Yes

'''

import os
import argparse
import numpy as np
import random
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import optimizers
from rdkit import Chem
from model.config import read_config, print_all_hyper_parameters
from utils.dircheck import dircheck
from utils.distance import euc_dist
from utils.Utils import *
from model.model import GAT
from model.prepare_data import set_data_list, get_batch_data
from model.metrics import *
from utils.get_time import get_time

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


def get_arguments():
    parser = argparse.ArgumentParser(description='Model Operation')

    parser.add_argument('-c', '--config', dest='config', type=str, required=True)
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True)
    parser.add_argument('-p', '--pocket', dest='pocket', type=str, help='Only for Prediction mode')
    parser.add_argument('-l', '--ligand', dest='ligand', nargs=2, help='Only for Prediction mode, index 0: fpath, idex 1: SMILES')
    
    return parser.parse_args()


def train_GAT(f_config, transfer='Autoencoder'):
    start_time = get_time()
    args = read_config(f_config) #transfer, f_train_list, f_valid_list, pocket_dir, ligand_dir, max_poc_node, max_mol_node, epochs, learning_rate, batch_size, l2_param, checkpoint, prefix
    ckpt = args['checkpoint']
    checkpoint_save = os.path.join(ckpt, 'params')
    dircheck(ckpt)

    LOG = open(f'{ckpt}/train.log', 'w')
    print_all_hyper_parameters(args, LOG)
    
    # Data loading 
    trainset = set_data_list(
        args['train_list'], args['pocket_dir'], args['ligand_dir']
    )
    validset = set_data_list(
        args['valid_list'], args['pocket_dir'], args['ligand_dir']
    )
    random.shuffle(trainset)

    LOG.write(f'Trainset: {len(trainset)}\n')
    LOG.write(f'Validset: {len(validset)}\n')

    tr_total_batch = int(len(trainset) / args['batch_size'])
    va_total_batch = int(len(validset) / args['batch_size'])

    # Weights loading 
    if args['transfer'] == 'GCN':
        params = os.path.join(ckpt, 'params/params.npy')
        W_poc_layer, b_poc_layer, W_poc_att_self, W_poc_att_neighbor, W_lig_layer, b_lig_layer, W_lig_att_self, W_lig_att_neighbor, W_inter, b_inter, W_out, b_out = np.load(params, allow_pickle=True)
        
        old_params = os.path.join(ckpt, 'GCN/old_params.npy')
        os.system(f'mv {params} {old_params}')
        
    elif args['transfer'] == 'None':
        os.makedirs(checkpoint_save)
        W_poc_layer = [
            [None, None, None],
            [None, None, None]
        ]
        b_poc_layer = [
            [None, None, None],
            [None, None, None]
        ]
        W_poc_att_self = [
            [None, None, None],
            [None, None, None]
        ]
        W_poc_att_neighbor = [
            [None, None, None],
            [None, None, None]
        ]
        W_lig_layer = [
            [None, None, None],
            [None, None, None]
        ]
        b_lig_layer = [
            [None, None, None],
            [None, None, None]
        ]
        W_lig_att_self = [
            [None, None, None],
            [None, None, None]
        ]
        W_lig_att_neighbor = [
            [None, None, None],
            [None, None, None]
        ]        
        W_inter = None
        b_inter = None
        W_out = None
        b_out = None
   
    # Define model feature dimensions
    poc_check_data = np.load(trainset[0][0], allow_pickle=True)
    lig_check_data = np.load(trainset[0][1], allow_pickle=True)

    n_poc_feature = poc_check_data[0].shape[-1]
    n_atom_feature = lig_check_data[0].shape[-1]
    n_bond_feature = lig_check_data[1].shape[-1]

    # Define a model
    model = GAT(
        n_poc_feature=n_poc_feature,
        n_atom_feature=n_atom_feature,
        n_bond_feature=n_bond_feature,
        batch_size=args['batch_size'], 
        W_poc_layer=W_poc_layer,
        b_poc_layer=b_poc_layer,
        W_poc_att_self=W_poc_att_self,
        W_poc_att_neighbor=W_poc_att_neighbor,
        W_lig_layer=W_lig_layer,
        b_lig_layer=b_lig_layer,
        W_lig_att_self=W_lig_att_self,
        W_lig_att_neighbor=W_lig_att_neighbor,
        W_inter=W_inter,
        b_inter=b_inter,
        W_out=W_out,
        b_out=b_out
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args['learning_rate'],
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True
    )
    #optimizer = optimizers.RMSprop(learning_rate = args['learning_rate'])
    optimizer = optimizers.RMSprop(learning_rate = lr_schedule)
    
    O_loss = open(f'{ckpt}/loss.csv', 'w')
    O_loss.write('Epoch,Train,Valid\n')
    O_acc = open(f'{ckpt}/train_accuracy.csv', 'w')
    O_acc.write('Epoch,Train,Valid\n')

    loss_dict = defaultdict(list)
    acc_dict = defaultdict(list)
    for epoch in range(1, args['epochs']+1):
        # Training
        for batch_data in get_batch_data(trainset, args['batch_size']):
            poc_feat, poc_adj, poc_mask, poc_d_score, pids, res_names = batch_data[:6]
            lig_atom_feat, lig_bond_feat, lig_atom_adj, lig_bond_adj = batch_data[6:10]
            lig_mask, label, smiles, cids = batch_data[10:14]

            # Model call and calculate gradients
            with tf.GradientTape() as tape:
                class_score, class_prob, classification, _, _, _, _, _ = model(
                    (poc_feat, poc_adj, poc_d_score, poc_mask,
                     lig_atom_feat, lig_bond_feat, lig_atom_adj, lig_bond_adj,
                     lig_mask),
                    training=True
                )
                tr_loss = model.loss(class_score, label)

                if args['l2_params'] != 0.:
                    for param in model.trainable_variables:
                        tr_loss = tf.add(
                            tr_loss, args['l2_params']*tf.nn.l2_loss(param)
                        )
            
            # Weight update
            grads = tape.gradient(tr_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Metric 
            #tr_avg_loss += tr_loss
            #tr_scores_list.extend(tr_class_score)
            #tr_labels_list.extend(tr_labels)

        # Validation
        tr_avg_loss = 0.
        va_avg_loss = 0.
        tr_scores_list = []
        tr_labels_list = []
        va_scores_list = []
        va_labels_list = []
        for batch_data in get_batch_data(trainset, args['batch_size']):
            poc_feat, poc_adj, poc_mask, poc_d_score, pids, res_names = batch_data[:6]
            lig_atom_feat, lig_bond_feat, lig_atom_adj, lig_bond_adj = batch_data[6:10]
            lig_mask, label, smiles, cids = batch_data[10:14]

            # Model call
            class_score, _, _, _, _, _, _, _ = model(
                (poc_feat, poc_adj, poc_d_score, poc_mask,
                 lig_atom_feat, lig_bond_feat, lig_atom_adj, lig_bond_adj,
                 lig_mask),
                training=False
            )
            tr_loss = model.loss(class_score, label)
            
            # Metric 
            #tr_avg_loss += tr_loss
            tr_avg_loss += np.sum(tr_loss)
            tr_scores_list.extend(class_score)
            tr_labels_list.extend(label)

        for batch_data in get_batch_data(validset, args['batch_size']):
            # Mini batch data load
            poc_feat, poc_adj, poc_mask, poc_d_score, pids, res_names = batch_data[:6]
            lig_atom_feat, lig_bond_feat, lig_atom_adj, lig_bond_adj = batch_data[6:10]
            lig_mask, label, smiles, cids = batch_data[10:14]

            # Model call
            class_score, _, _, _, _, _, _, _ = model(
                (poc_feat, poc_adj, poc_d_score, poc_mask,
                 lig_atom_feat, lig_bond_feat, lig_atom_adj, lig_bond_adj,
                 lig_mask),
                training=False
            )
            va_loss = model.loss(class_score, label)
            
            # Metric
            #va_avg_loss += va_loss
            va_avg_loss += np.sum(va_loss)
            va_scores_list.extend(class_score)
            va_labels_list.extend(label)

        # Save metrics
        tr_avg_loss = tr_avg_loss / (tr_total_batch * args['batch_size'])
        va_avg_loss = va_avg_loss / (va_total_batch * args['batch_size'])
        tr_avg_loss = round(float(tr_avg_loss), 4)
        va_avg_loss = round(float(va_avg_loss), 4)
        loss_dict['tr'].append(tr_avg_loss)
        loss_dict['va'].append(va_avg_loss)

        tr_scores_list = np.array(tr_scores_list)
        tr_labels_list = np.array(tr_labels_list)
        va_scores_list = np.array(va_scores_list)
        va_labels_list = np.array(va_labels_list)

        tr_acc = accuracy(tr_scores_list, tr_labels_list)
        va_acc = accuracy(va_scores_list, va_labels_list)
        tr_acc = round(float(tr_acc), 4)
        va_acc = round(float(va_acc), 4)
        acc_dict['tr'].append(tr_acc)
        acc_dict['va'].append(va_acc)

        tr_auc = get_auc(tr_scores_list, tr_labels_list)
        va_auc = get_auc(va_scores_list, va_labels_list)
        tr_auc = round(float(tr_auc), 4)
        va_auc = round(float(va_auc), 4)

        O_loss.write(
            f'[Epoch {epoch}],{tr_avg_loss},{va_avg_loss}\n'
        )
        O_acc.write(
            f'[Epoch {epoch}],{tr_acc},{va_acc}\n'
        )
        
        print(f'[Epoch {epoch}], {get_time()}')
        print(f'Train loss: {tr_avg_loss}, Valid loss: {va_avg_loss}')
        print(f'Train acc: {tr_acc}, Valid acc: {va_acc}')
        print(f'Train auc: {tr_auc}, Valid auc: {va_auc}')

        # Save parameters every 5 epochs
        if epoch % 5 == 0:
            params = model.get_params()
            np.save(f'{checkpoint_save}/params.npy', params)
            np.save(f'{checkpoint_save}/params_{epoch}.npy', params)
            loss_plot(loss_dict, ckpt, args['prefix'], args['epochs'])
            accuracy_plot(acc_dict, ckpt, args['prefix'], args['epochs'])

    O_loss.close()
    O_acc.close()
    
    loss_plot(loss_dict, ckpt, args['prefix'], args['epochs'])
    accuracy_plot(acc_dict, ckpt, args['prefix'], args['epochs'])

    LOG.write(f'Start: {start_time}\n')
    LOG.write(f'End: {get_time()}\n')

    LOG.close()

def eval_GAT(f_config):
    start_time = get_time()
    args = read_config(f_config) #transfer, f_train_list, f_valid_list, pocket_dir, ligand_dir, max_poc_node, max_mol_node, epochs, learning_rate, batch_size, l2_param, checkpoint, prefix
    ckpt = args['checkpoint']
    prefix = args['prefix']
    checkpoint_load = os.path.join(ckpt, 'params')
    dircheck(ckpt)

    # Data loading 
    testset = set_data_list(
        args['test_list'], args['pocket_dir'], args['ligand_dir']
    )
    
    # Weights loading
    params = os.path.join(checkpoint_load, 'params.npy')
    W_poc_layer, b_poc_layer, W_poc_att_self, W_poc_att_neighbor, W_lig_layer, b_lig_layer, W_lig_att_self, W_lig_att_neighbor, W_inter, b_inter, W_out, b_out = np.load(params, allow_pickle=True)

    # Model assessment
    poc_check_data = np.load(testset[0][0], allow_pickle=True)
    lig_check_data = np.load(testset[0][1], allow_pickle=True)

    n_poc_feature = poc_check_data[0].shape[-1]
    n_atom_feature = lig_check_data[0].shape[-1]
    n_bond_feature = lig_check_data[1].shape[-1]
    
    model = GAT(
        n_poc_feature=n_poc_feature,
        n_atom_feature=n_atom_feature,
        n_bond_feature=n_bond_feature,
        batch_size=args['batch_size'], 
        W_poc_layer=W_poc_layer,
        b_poc_layer=b_poc_layer,
        W_poc_att_self=W_poc_att_self,
        W_poc_att_neighbor=W_poc_att_neighbor,
        W_lig_layer=W_lig_layer,
        b_lig_layer=b_lig_layer,
        W_lig_att_self=W_lig_att_self,
        W_lig_att_neighbor=W_lig_att_neighbor,
        W_inter=W_inter,
        b_inter=b_inter,
        W_out=W_out,
        b_out=b_out
    )
    
    class_scores = []
    class_probs = []
    classifications = []
    labels = []
    PIDs = []
    CIDs = []
    smi_list = []

    for batch_data in get_batch_data(testset, args['batch_size']):
        # Mini batch data load
        poc_feat, poc_adj, poc_mask, poc_d_score, pids, res_names = batch_data[:6]
        lig_atom_feat, lig_bond_feat, lig_atom_adj, lig_bond_adj = batch_data[6:10]
        lig_mask, label, smiles, cids = batch_data[10:14]

        # Model call    
        class_score, class_prob, classification, _, _, _, _, _ = model(
            (poc_feat, poc_adj, poc_d_score, poc_mask,
             lig_atom_feat, lig_bond_feat, lig_atom_adj, lig_bond_adj,
             lig_mask),
            training=False
        )

        class_scores.extend(class_score)
        class_probs.extend(class_prob)
        classifications.extend(classification)
        labels.extend(label)
        PIDs.extend(pids)
        CIDs.extend(cids)
        smi_list.extend(smiles)

    class_scores = np.array(class_scores)
    class_probs = np.array(class_probs)
    labels = np.array(labels)

    TP, TN, FP, FN, acc, precision, recall, f1 = Stats(class_scores, labels, ckpt, prefix)
    auc = ROC_curve(class_probs, labels, ckpt, prefix)

    filename = os.path.join(ckpt, f'{prefix}_stats.txt')
    O_eval = open(filename, 'w')
    O_eval.write(
        f'True Positive: {TP}\n' +
        f'True Negative: {TN}\n' +
        f'False Positive: {FP}\n' +
        f'False Negative: {FN}\n' +
        f'Accuracy: {acc}\n' +
        f'AUC: {auc}\n' +
        f'Precision: {precision}\n' +
        f'Recall: {recall}\n' +
        f'F1 score: {f1}\n' 
    )
    O_eval.close()

    predict_fpath = os.path.join(ckpt, f'{prefix}_prediction.csv')
    prediction(predict_fpath, PIDs, CIDs, classifications, class_probs, smi_list, labels)

    print(f'Start: {start_time}')
    print(f'End: {get_time()}')


def predict_GAT(op_args):
    f_config = op_args.config

    start_time = get_time()
    args = read_config(f_config) #transfer, f_train_list, f_valid_list, pocket_dir, ligand_dir, max_poc_node, max_mol_node, epochs, learning_rate, batch_size, l2_param, checkpoint, prefix
    ckpt = args['checkpoint']
    prefix = args['prefix']
    checkpoint_load = os.path.join(ckpt, 'params')
    dircheck(ckpt)

    # Data loading 
    if op_args.pocket and op_args.ligand:
        # args.pocket = pocket file name
        assert len(op_args.ligand) == 2
    
        f_poc = os.path.join(args['pocket_dir'], op_args.pocket)
        f_lig = os.path.join(args['ligand_dir'], op_args.ligand[0])
        mol = Chem.MolFromSmiles(op_args.ligand[1])
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        testset = [[f_poc, f_lig, 1, smiles]]
    else:
        testset = set_data_list(
            args['test_list'], args['pocket_dir'], args['ligand_dir']
        )
    
    # Weights loading
    params = os.path.join(checkpoint_load, 'params.npy')
    W_poc_layer, b_poc_layer, W_poc_att_self, W_poc_att_neighbor, W_lig_layer, b_lig_layer, W_lig_att_self, W_lig_att_neighbor, W_inter, b_inter, W_out, b_out = np.load(params, allow_pickle=True)

    # Model assessment
    poc_check_data = np.load(testset[0][0], allow_pickle=True)
    lig_check_data = np.load(testset[0][1], allow_pickle=True)

    n_poc_feature = poc_check_data[0].shape[-1]
    n_atom_feature = lig_check_data[0].shape[-1]
    n_bond_feature = lig_check_data[1].shape[-1]
    
    model = GAT(
        n_poc_feature=n_poc_feature,
        n_atom_feature=n_atom_feature,
        n_bond_feature=n_bond_feature,
        batch_size=args['batch_size'], 
        W_poc_layer=W_poc_layer,
        b_poc_layer=b_poc_layer,
        W_poc_att_self=W_poc_att_self,
        W_poc_att_neighbor=W_poc_att_neighbor,
        W_lig_layer=W_lig_layer,
        b_lig_layer=b_lig_layer,
        W_lig_att_self=W_lig_att_self,
        W_lig_att_neighbor=W_lig_att_neighbor,
        W_inter=W_inter,
        b_inter=b_inter,
        W_out=W_out,
        b_out=b_out
    )
    
    class_probs = []
    classifications = []
    PIDs = []
    CIDs = []
    smi_list = []
    for batch_data in get_batch_data(testset, args['batch_size']):
        # Mini batch data load
        poc_feat, poc_adj, poc_mask, poc_d_score, pids, res_names = batch_data[:6]
        lig_atom_feat, lig_bond_feat, lig_atom_adj, lig_bond_adj = batch_data[6:10]
        lig_mask, label, smiles, cids = batch_data[10:14]

        # Model call    
        class_score, class_prob, classification, _, _, _, _, _ = model(
            (poc_feat, poc_adj, poc_d_score, poc_mask,
             lig_atom_feat, lig_bond_feat, lig_atom_adj, lig_bond_adj,
             lig_mask),
            training=False
        )

        class_probs.extend(class_prob)
        classifications.extend(classification)
        PIDs.extend(pids)
        CIDs.extend(cids)
        smi_list.extend(smiles)
    
    predict_fpath = os.path.join(ckpt, f'{prefix}_prediction.csv')
    prediction(predict_fpath, PIDs, CIDs, classification, class_probs, smi_list)

    print(f'Start: {start_time}')
    print(f'End: {get_time()}')


if __name__ == '__main__':
    op_args = get_arguments()

    if op_args.mode == 'train':
        train_GAT(op_args.config, transfer=False)
    elif op_args.mode == 'eval':
        eval_GAT(op_args.config)
    elif op_args.mode == 'predict':
        predict_GAT(op_args)
