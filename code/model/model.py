#!/usr/bin/env python
# coding: utf-8
'''
Last update: 21.01.21. KS.Kwon

'''

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layer import GraphConv, GraphConv_mol, Dropout_layer, Output_layer


class GAT(keras.models.Model):
    max_poc_node = 100
    max_mol_node = 60

    def __init__(self, n_poc_feature, n_atom_feature, n_bond_feature, batch_size,
                 W_poc_layer, b_poc_layer, W_poc_att_self, W_poc_att_neighbor,
                 W_lig_layer, b_lig_layer, W_lig_att_self, W_lig_att_neighbor,
                 W_inter, b_inter, W_out, b_out, **kwargs):
        super().__init__(**kwargs)

        self._hidden_features = [300, 200]

        self.Pocket_GAT1 = GraphConv(           
            max_poc_node=GAT.max_poc_node, 
            n_feature=n_poc_feature, 
            n_hidden=self._hidden_features[0],
            batch_size=batch_size,
            W_layer=W_poc_layer[0],
            b_layer=b_poc_layer[0],
            W_att_self=W_poc_att_self[0],
            W_att_neighbor=W_poc_att_neighbor[0]
        )
        self.Pocket_GAT2 = GraphConv(
            max_poc_node=GAT.max_poc_node, 
            n_feature=self._hidden_features[0] * 3, 
            n_hidden=self._hidden_features[1],
            batch_size=batch_size,
            W_layer=W_poc_layer[1],
            b_layer=b_poc_layer[1],
            W_att_self=W_poc_att_self[1],
            W_att_neighbor=W_poc_att_neighbor[1]
        )
        self.Ligand_GAT1 = GraphConv_mol(
            max_mol_node=GAT.max_mol_node, 
            n_atom_feature=n_atom_feature,
            n_bond_feature=n_bond_feature,
            n_hidden=self._hidden_features[0],  
            batch_size=batch_size,
            W_layer=W_lig_layer[0], 
            b_layer=b_lig_layer[0],
            W_att_self=W_lig_att_self[0],
            W_att_neighbor=W_lig_att_neighbor[0]
        )
        self.Ligand_GAT2 = GraphConv_mol(
            max_mol_node=GAT.max_mol_node, 
            n_atom_feature=self._hidden_features[0] * 3,
            n_bond_feature=n_bond_feature,
            n_hidden=self._hidden_features[1],
            batch_size=batch_size,
            W_layer=W_lig_layer[1], 
            b_layer=b_lig_layer[1],
            W_att_self=W_lig_att_self[1],
            W_att_neighbor=W_lig_att_neighbor[1],
            add_bond_feat=False
        )
        self.Interaction_layer = Dropout_layer(
            n_input=6*self._hidden_features[1],
            n_output=100, W=W_inter, b=b_inter
        )
        self.Output_layer = Output_layer(
            n_input=100, n_output=2, W=W_out, b=b_out
        )

    def call(self, inputs, training=False):
        M_poc_feature, M_poc_adj, poc_d_score, poc_mask, M_atom_feature, M_bond_feature, M_atom_adj, M_bond_adj, mol_mask = inputs

        M_poc_hidden1 = self.Pocket_GAT1(
            (M_poc_feature, M_poc_adj, poc_mask),
            training=training
        )
        M_poc_hidden2 = self.Pocket_GAT2(
            (M_poc_hidden1, M_poc_adj, poc_mask),
            training=training
        )

        M_lig_hidden1 = self.Ligand_GAT1(
            (M_atom_feature, M_bond_feature, M_atom_adj, M_bond_adj, mol_mask),
            training=training
        )
        M_lig_hidden2 = self.Ligand_GAT2(
            (M_lig_hidden1, M_bond_feature, M_atom_adj, M_bond_adj, mol_mask),
            training=training
        )

        poc_FP = self._convToFP(M_poc_hidden2, poc_mask)
        lig_FP = self._convToFP(M_lig_hidden2, mol_mask)

        inter_input = tf.concat([poc_FP, lig_FP], axis=1)
        inter_output = self.Interaction_layer(inter_input, training=training)
        class_score, class_prob, classification = self.Output_layer(inter_output)

        return class_score, class_prob, classification, inter_output, poc_FP, M_poc_hidden2, lig_FP, M_lig_hidden2
    
    def loss(self, inputs, labels):    
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=inputs)

    def _convToFP(self, matrix, mask):
        n = tf.reduce_sum(mask, axis=1)
        vector = tf.reduce_sum(matrix, axis=1)
        FP = vector / tf.expand_dims(n, axis=-1)

        return FP

    def get_params(self):
        W_poc_layer1, b_poc_layer1, W_poc_att_self1, W_poc_att_neighbor1 = self.Pocket_GAT1.get_params()
        W_poc_layer2, b_poc_layer2, W_poc_att_self2, W_poc_att_neighbor2 = self.Pocket_GAT2.get_params()
        
        W_lig_layer1, b_lig_layer1, W_lig_att_self1, W_lig_att_neighbor1 = self.Ligand_GAT1.get_params()
        W_lig_layer2, b_lig_layer2, W_lig_att_self2, W_lig_att_neighbor2 = self.Ligand_GAT2.get_params()

        return np.array([
            [W_poc_layer1, W_poc_layer2],
            [b_poc_layer1, b_poc_layer2],
            [W_poc_att_self1, W_poc_att_self2],
            [W_poc_att_neighbor1, W_poc_att_neighbor2],
            [W_lig_layer1, W_lig_layer2],
            [b_lig_layer1, b_lig_layer2],
            [W_lig_att_self1, W_lig_att_self2],
            [W_lig_att_neighbor1, W_lig_att_neighbor2],
            self.Interaction_layer.W, self.Interaction_layer.b,
            self.Output_layer.W, self.Output_layer.b
        ], dtype=object)
