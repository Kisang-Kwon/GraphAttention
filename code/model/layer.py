#!/usr/bin/env python
# coding: utf-8
'''
Last update: 21.07.21. by KS.Kwon

'''

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from .inits import glorot, He, zeros


class Pocket_attention(layers.Layer):
    def __init__(self, max_poc_node, n_feature, n_hidden, batch_size,
                 W_layer, b_layer, W_att_self, W_att_neighbor, dropout=0.5,
                 activation=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)

        self.max_poc_node = max_poc_node
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.dropout = dropout
        self.activation = activation

        if W_layer is None:
            self.W_layer = tf.Variable(
                initial_value=He([self.n_feature, self.n_hidden])
            )
        else:
            self.W_layer = tf.Variable(initial_value=W_layer)

        if b_layer is None:
            self.b_layer = tf.Variable(
                initial_value=zeros([self.n_hidden, ])
            )
        else:
            self.b_layer = tf.Variable(initial_value=b_layer)

        if W_att_self is None:
            self.W_att_self = tf.Variable(initial_value=He([self.n_hidden, 1]))
        else:
            self.W_att_self = tf.Variable(initial_value=W_att_self)
        
        if W_att_neighbor is None:
            self.W_att_neighbor = tf.Variable(
                initial_value=He([self.n_hidden, 1]))
        else:
            self.W_att_neighbor = tf.Variable(initial_value=W_att_neighbor)

    def call(self, M_features, M_adjacency, mask, training):

        # (Nodes x Hidden features)
        hidden_features = tf.matmul(M_features, self.W_layer)

        ''' Attention '''
        attn_for_self = tf.matmul(hidden_features, self.W_att_self)
        attn_for_neighbor = tf.matmul(hidden_features, self.W_att_neighbor)

        attention = attn_for_self + \
            tf.transpose(attn_for_neighbor, perm=[0, 2, 1])
        attention = tf.nn.leaky_relu(attention)

        att_mask = -10e9 * (1.0 - M_adjacency)
        attention += att_mask

        ''' softmax to get attention coefficients (Equation 3) '''
        attention = tf.nn.softmax(attention)

        ''' Dropout to attention & hidden features '''
        if training:
            attention = tf.nn.dropout(attention, self.dropout)
            hidden_features = tf.nn.dropout(hidden_features, self.dropout)

        h_prime = tf.matmul(attention, hidden_features) + \
            tf.expand_dims(self.b_layer, axis=0)
        h_prime = h_prime * tf.expand_dims(mask, axis=-1)

        return h_prime


class GraphAttention(layers.Layer):
    def __init__(self, max_poc_node, n_feature, n_hidden, batch_size,
                 W_layer, b_layer, W_att_self, W_att_neighbor, dropout=0.5, 
                 activation=tf.nn.relu, attn_out='concat', **kwargs):
        super().__init__(**kwargs)

        self.max_poc_node = max_poc_node
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.dropout = dropout
        self.activation = activation
        self.attn_out = attn_out

        params = {
            'max_poc_node': max_poc_node, 'n_feature': n_feature,
            'n_hidden': n_hidden, 'batch_size': batch_size, 'dropout': dropout
        }

        self.attention1 = Pocket_attention(
            W_layer=W_layer[0],
            b_layer=b_layer[0],
            W_att_self=W_att_self[0],
            W_att_neighbor=W_att_neighbor[0],
            **params
        )
        self.attention2 = Pocket_attention(
            W_layer=W_layer[1],
            b_layer=b_layer[1],
            W_att_self=W_att_self[1],
            W_att_neighbor=W_att_neighbor[1],
            **params
        )
        self.attention3 = Pocket_attention(
            W_layer=W_layer[2],
            b_layer=b_layer[2],
            W_att_self=W_att_self[2],
            W_att_neighbor=W_att_neighbor[2],
            **params
        )

    def call(self, inputs, training=False):
        M_features, M_adjacency, mask = inputs

        att1 = self.attention1(M_features, M_adjacency, mask, training)
        att2 = self.attention2(M_features, M_adjacency, mask, training)
        att3 = self.attention3(M_features, M_adjacency, mask, training)

        if self.attn_out == 'concat':
            output = tf.concat([att1, att2, att3], axis=2)
        else:
            output = tf.reduce_mean([att1, att2, att3], axis=1)

        output = self.activation(output)

        return output

    def get_params(self):
        return [
            [self.attention1.W_layer,
            self.attention2.W_layer,
            self.attention3.W_layer],

            [self.attention1.b_layer,
            self.attention2.b_layer,
            self.attention3.b_layer],

            [self.attention1.W_att_self,
            self.attention2.W_att_self,
            self.attention3.W_att_self],

            [self.attention1.W_att_neighbor,
            self.attention2.W_att_neighbor,
            self.attention3.W_att_neighbor]
        ]


class Ligand_attention(layers.Layer):
    def __init__(self, max_mol_node, n_atom_feature, n_bond_feature,
                 n_hidden, batch_size, W_layer, b_layer, W_att_self,
                 W_att_neighbor, dropout=0.3, add_bond_feat=True, **kwargs):
        super().__init__(**kwargs)

        self.max_mol_node = max_mol_node
        self.n_atom_feature = n_atom_feature
        self.n_bond_feature = n_bond_feature
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.add_bond_feat = add_bond_feat
        self.dropout = dropout

        if W_layer is None:
            if self.add_bond_feat:
                self.W_layer = tf.Variable(
                    initial_value=He(
                        [self.n_atom_feature + self.n_bond_feature, self.n_hidden]
                    )
                )
            else:
                self.W_layer = tf.Variable(
                    initial_value=He([self.n_atom_feature, self.n_hidden])
                )
        else:
            self.W_layer = tf.Variable(initial_value=W_layer)

        if b_layer is None:
            self.b_layer = tf.Variable(
                initial_value=zeros([self.n_hidden, ])
            )
        else:
            self.b_layer = tf.Variable(initial_value=b_layer)

        if W_att_self is None:
            self.W_att_self = tf.Variable(initial_value=He([self.n_hidden, 1]))
        else:
            self.W_att_self = tf.Variable(initial_value=W_att_self)

        if W_att_neighbor is None:
            self.W_att_neighbor = tf.Variable(
                initial_value=He([self.n_hidden, 1]))
        else:
            self.W_att_neighbor = tf.Variable(initial_value=W_att_neighbor)

    def call(self, M_atom_features, M_bond_features, M_atom_adjacency,
             M_bond_adjacency, mask, training):

        if self.add_bond_feat:
            bond_features = tf.matmul(M_bond_adjacency, M_bond_features)
            M_features = tf.concat([M_atom_features, bond_features], 2)
            hidden_features = tf.matmul(M_features, self.W_layer)
        else:
            M_features = M_atom_features
            hidden_features = tf.matmul(M_features, self.W_layer)

        ''' Attention '''
        attn_for_self = tf.matmul(hidden_features, self.W_att_self)
        attn_for_neighbor = tf.matmul(hidden_features, self.W_att_neighbor)

        attention = attn_for_self + \
            tf.transpose(attn_for_neighbor, perm=[0, 2, 1])
        attention = tf.nn.leaky_relu(attention)

        att_mask = -10e9 * (1.0 - M_atom_adjacency)
        attention += att_mask

        ''' softmax to get attention coefficients (Equation 3) '''
        attention = tf.nn.softmax(attention)

        ''' Dropout to attention & hidden features '''
        if training:
            attention = tf.nn.dropout(attention, self.dropout)
            hidden_features = tf.nn.dropout(hidden_features, self.dropout)

        h_prime = tf.matmul(attention, hidden_features) + \
            tf.expand_dims(self.b_layer, 0)
        h_prime = h_prime * tf.expand_dims(mask, -1)

        return h_prime


class GraphAttention_mol(layers.Layer):
    def __init__(self, max_mol_node, n_atom_feature, n_bond_feature, n_hidden,
                 batch_size, W_layer, b_layer, W_att_self, W_att_neighbor,
                 dropout=0.5, attn_out='concat', activation=tf.nn.relu,
                 add_bond_feat=True, **kwargs):
        super().__init__(**kwargs)

        self.max_mol_node = max_mol_node
        self.n_atom_feature = n_atom_feature
        self.n_bond_feature = n_bond_feature
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.dropout = dropout
        self.activation = activation
        self.add_bond_feat = add_bond_feat
        self.attn_out = attn_out

        params = {
            'max_mol_node': max_mol_node, 'n_atom_feature': n_atom_feature,
            'n_bond_feature': n_bond_feature, 'n_hidden': n_hidden,
            'batch_size': batch_size, 'add_bond_feat': add_bond_feat
        }

        self.attention1 = Ligand_attention(
            W_layer=W_layer[0], b_layer=b_layer[0], 
            W_att_self=W_att_self[0], W_att_neighbor=W_att_neighbor[0],
            **params
        )
        self.attention2 = Ligand_attention(
            W_layer=W_layer[1], b_layer=b_layer[1], 
            W_att_self=W_att_self[1], W_att_neighbor=W_att_neighbor[1],
            **params
        )
        self.attention3 = Ligand_attention(
            W_layer=W_layer[2], b_layer=b_layer[2], 
            W_att_self=W_att_self[2], W_att_neighbor=W_att_neighbor[2],
            **params
        )

    def call(self, inputs, training=False):
        M_atom_features, M_bond_features, M_atom_adjacency, M_bond_adjacency, mask = inputs

        att1 = self.attention1(
            M_atom_features, M_bond_features, M_atom_adjacency,
            M_bond_adjacency, mask, training
        )
        att2 = self.attention2(
            M_atom_features, M_bond_features, M_atom_adjacency,
            M_bond_adjacency, mask, training
        )
        att3 = self.attention3(
            M_atom_features, M_bond_features, M_atom_adjacency,
            M_bond_adjacency, mask, training
        )

        if self.attn_out == 'concat':
            output = tf.concat([att1, att2, att3], axis=2)
        else:
            output = tf.reduce_mean([att1, att2, att3], axis=1)
        
        output = self.activation(output)

        return output

    def get_params(self):
        return [
            [self.attention1.W_layer,
            self.attention2.W_layer,
            self.attention3.W_layer],

            [self.attention1.b_layer,
            self.attention2.b_layer,
            self.attention3.b_layer],

            [self.attention1.W_att_self,
            self.attention2.W_att_self,
            self.attention3.W_att_self],

            [self.attention1.W_att_neighbor,
            self.attention2.W_att_neighbor,
            self.attention3.W_att_neighbor]
        ]


class Interaction_layer(layers.Layer):
    def __init__(self, n_input, n_output, W=None, b=None, activation=tf.math.tanh, **kwargs):
        super().__init__(**kwargs)

        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation

        if W is None:
            self.W = tf.Variable(
                name='W_inter',
                initial_value=glorot([self.n_input, self.n_output])
            )
            if self.activation == tf.math.sigmoid:
                self.W = 4 * self.W
        else:
            self.W = W

        if b is None:
            self.b = tf.Variable(
                name='b_inter',
                initial_value=zeros([self.n_output, ])
            )
        else:
            self.b = b

    def call(self, inputs):
        linear_output = tf.matmul(inputs, self.W) + self.b
        out = self.activation(linear_output)

        return out


class Dropout_layer(layers.Layer):
    def __init__(self,
                 n_input,
                 n_output,
                 dropout=0.5,
                 W=None,
                 b=None,
                 activation=tf.nn.relu,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
        self.dropout = dropout

        if W is None:
            self.W = tf.Variable(
                name='W_inter',
                initial_value=He([self.n_input, self.n_output])
            )
        else:
            self.W = W

        if b is None:
            self.b = tf.Variable(
                name='b_inter',
                initial_value=zeros([self.n_output, ])
            )
        else:
            self.b = b

    def call(self, inputs, training=False):
        linear_output = tf.matmul(inputs, self.W) + self.b
        if training:
            inputs = tf.nn.dropout(linear_output, rate=self.dropout, seed=1)
        out = self.activation(linear_output)

        return out


class Output_layer(layers.Layer):
    def __init__(self,
                 n_input,
                 n_output,
                 W=None,
                 b=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_input = n_input
        self.n_output = n_output

        if W is None:
            self.W = tf.Variable(
                name='W_log',
                initial_value=zeros([self.n_input, self.n_output])
            )
        else:
            self.W = W

        if b is None:
            self.b = tf.Variable(
                name='b_log',
                initial_value=zeros([self.n_output, ])
            )
        else:
            self.b = b

    def call(self, inputs):
        y_probs = tf.nn.softmax(tf.matmul(inputs, self.W) + self.b)
        score = tf.matmul(inputs, self.W) + self.b
        y_pred = y_probs[:, 1] >= 0.5
        y_pred = tf.cast(y_pred, tf.int64)

        return score, y_probs, y_pred
