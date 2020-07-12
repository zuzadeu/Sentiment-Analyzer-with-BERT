# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 16:10:12 2020

@author: zuzan
"""

from model_experiment import ex
import itertools

batch_size_values =[128]
dense_1_nodes =[128]
dense_2_nodes = [128]
dense_3_nodes= [128]
dense_4_nodes = [128]
dropout_1_size = [0.1]
dropout_2_size= [0.2]
dropout_3_size = [0.3]
dropout_4_size = [0.2]
dropout_5_size = [0.1]
epochs_values = [30]


for epochs, batch_size, dense_1_nodes, dense_2_nodes, dense_3_nodes, dense_4_nodes, dropout_1_size, dropout_2_size, dropout_3_size, dropout_4_size, dropout_5_size in itertools.product(epochs_values, batch_size_values, dense_1_nodes, dense_2_nodes, dense_3_nodes, dense_4_nodes,dropout_1_size, dropout_2_size, dropout_3_size, dropout_4_size, dropout_5_size):
    ex.run(config_updates={'dropout_5_size': dropout_5_size, 'batch_size': batch_size, 'epochs': epochs,
                           'dense_1_nodes': dense_1_nodes, 'dense_2_nodes': dense_2_nodes,
                           'dense_3_nodes': dense_3_nodes, 'dense_4_nodes': dense_4_nodes,
                           'dropout_1_size': dropout_1_size, 'dropout_2_size': dropout_2_size,
                           'dropout_3_size': dropout_3_size, 'dropout_4_size': dropout_4_size
                           })
