# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:04:10 2020

@author: zuzan
"""
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import collections
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment()
ex.observers.append(MongoObserver(
    url='mongodb://192.168.1.2:27017'))
    #url='mongodb://mongo_user:mongo_password@localhost:27017/?authMechanism=SCRAM-SHA-1'))
    
def load_model():
    train_x = pickle.load(open("train_x.pkl", 'rb'))
    print(train_x)
    train_y = pickle.load(open("train_y.pkl", 'rb'))
    test_x = pickle.load(open("test_x.pkl", 'rb'))
    test_y = pickle.load(open("test_y.pkl", 'rb'))
    valid_x = pickle.load(open("valid_x.pkl", 'rb'))
    valid_y = pickle.load(open("valid_y.pkl", 'rb'))
    return train_x, train_y, test_x, test_y, valid_x, valid_y

def build_model(train_x, batch_size=128, dropout_rate=0.3,
                dense_1_nodes=256, dense_2_nodes=256, dense_3_nodes=256, dense_4_nodes=256,
                dropout_1_size=0.1, dropout_2_size=0.1, dropout_3_size=0.1, dropout_4_size=0.1):
    model = Sequential()
    
    # The Input Layer :
    model.add(Dense(dense_1_nodes, kernel_initializer='normal',input_dim = train_x.shape[1], activation='relu'))
    model.add(Dropout(rate=dropout_1_size))
    
    # The Hidden Layers :
    model.add(Dense(dense_2_nodes, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_2_size))
    
    model.add(Dense(dense_3_nodes, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_3_size))
    
    model.add(Dense(dense_4_nodes, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_4_size))

    # The Output Layer :
    model.add(Dense(units = 1, activation="sigmoid"))
    model.add(Dropout(dropout_rate))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics = ['mse', 'mae'])
    print(model.summary)
    return model


def model_testing(test_x, test_y, model, history):
    pred_y = model.predict(test_x)
    np.savetxt("pred_y.csv", pred_y, delimiter=",")
    mae = mean_absolute_error(pred_y, test_y)
    mse = mean_squared_error(pred_y, test_y)
    print(f'MSE = {mse}')
    plt.plot(history.history['mse'])
    print(f'MAE = {mae}')
    plt.plot(history.history['mae'])
    plt.legend()
    plt.show()

    ex.log_scalar("MAE", mae)
    ex.log_scalar("MSE", mse)


@ex.config
def cfg():
    dropout_rate=0.3 
    epochs=30 
    batch_size=128
    dense_1_nodes=256 
    dense_2_nodes=256 
    dense_3_nodes=256 
    dense_4_nodes=256
    dropout_1_size=0.1 
    dropout_2_size=0.1 
    dropout_3_size=0.1 
    dropout_4_size=0.1
#sacred: epochs, dropout-rate, batch size, 

def model_training(train_x, train_y, valid_x, valid_y, dropout_rate, epochs, batch_size,
                   dense_1_nodes, dense_2_nodes, dense_3_nodes, dense_4_nodes,
                dropout_1_size, dropout_2_size, dropout_3_size, dropout_4_size):
    model = build_model(train_x, batch_size, dropout_rate,
                         dense_1_nodes, dense_2_nodes, dense_3_nodes, dense_4_nodes,
                dropout_1_size, dropout_2_size, dropout_3_size, dropout_4_size)
    history = model.fit(train_x.values, train_y.values, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(valid_x.values, valid_y.values))
    model.save('model.h5')
    return model, history

@ex.automain
def main(dropout_rate, epochs, batch_size,
         dense_1_nodes, dense_2_nodes, dense_3_nodes, dense_4_nodes,
         dropout_1_size, dropout_2_size, dropout_3_size, dropout_4_size):
    
    train_x, train_y, test_x, test_y, valid_x, valid_y = load_model()
    model, history = model_training(train_x, train_y, valid_x, valid_y, dropout_rate=dropout_rate, epochs=epochs, batch_size=batch_size,
                                    dense_1_nodes=dense_1_nodes, dense_2_nodes=dense_2_nodes, dense_3_nodes=dense_3_nodes, dense_4_nodes=dense_4_nodes,
                                    dropout_1_size=dropout_1_size, dropout_2_size=dropout_2_size, dropout_3_size=dropout_3_size, dropout_4_size=dropout_4_size)
    model_testing(test_x, test_y, model, history)
