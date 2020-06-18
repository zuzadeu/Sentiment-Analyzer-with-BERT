# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 16:10:12 2020

@author: zuzan
"""

from model_experiment import ex
import itertools



#0.3, 30, 256 - 0.2845, 0.1005
#0.1, 50, 512 - 0.1530, 0.0313
#0.06 60, 256 - 0.1303, 0.0304


#"batch_size":256
#"dense_1_nodes":32
#"dense_2_nodes":32
#"dense_3_nodes":32
#"dense_4_nodes":128
#"dropout_1_size":0.06
#"dropout_2_size":0.2
#"dropout_3_size":0.3
#"dropout_4_size":0.2
#"dropout_rate":0.06
#"epochs":45

dense_1_nodes=[32, 64, 128, 256, 512, 1024] 
dense_2_nodes=[32, 64, 128, 256, 512, 1024] 
dense_3_nodes=[32, 64, 128, 256, 512, 1024] 
dense_4_nodes=[32, 64, 128, 256, 512, 1024] 
dropout_1_size=[0.06, 0.08, 0.1, 0.2, 0.3] 
dropout_2_size=[0.06, 0.08, 0.1, 0.2, 0.3] 
dropout_3_size=[0.06, 0.08, 0.1, 0.2, 0.3] 
dropout_4_size=[0.06, 0.08, 0.1, 0.2, 0.3] 
dropout_rate_values = [0.06, 0.08]
epochs_values = [45, 50, 55, 60, 65]
batch_size_values = [256, 512]

for dropout_rate, epochs, batch_size, dense_1_nodes, dense_2_nodes, dense_3_nodes, dense_4_nodes, dropout_1_size, dropout_2_size, dropout_3_size, dropout_4_size in itertools.product(dropout_rate_values, epochs_values, batch_size_values, dense_1_nodes, dense_2_nodes, dense_3_nodes, dense_4_nodes,dropout_1_size, dropout_2_size, dropout_3_size, dropout_4_size):
    ex.run(config_updates={'dropout_rate': dropout_rate, 'batch_size': batch_size, 'epochs': epochs,
                           'dense_1_nodes': dense_1_nodes, 'dense_2_nodes': dense_2_nodes,
                           'dense_3_nodes': dense_3_nodes, 'dense_4_nodes': dense_4_nodes,
                           'dropout_1_size': dropout_1_size, 'dropout_2_size': dropout_2_size,
                           'dropout_3_size': dropout_3_size, 'dropout_4_size': dropout_4_size
                           })