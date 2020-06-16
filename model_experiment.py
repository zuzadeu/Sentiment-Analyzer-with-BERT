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


from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment()
ex.observers.append(MongoObserver(
    url='mongodb://localhost:27017'))
    #url='mongodb://mongo_user:mongo_password@localhost:27017/?authMechanism=SCRAM-SHA-1'))
    
def load_model():
    train_x = pickle.load(open("train_x.pkl", 'rb'))
    train_y = pickle.load(open("train_y.pkl", 'rb'))
    test_x = pickle.load(open("test_x.pkl", 'rb'))
    test_y = pickle.load(open("test_y.pkl", 'rb'))
    valid_x = pickle.load(open("valid_x.pkl", 'rb'))
    valid_y = pickle.load(open("valid_y.pkl", 'rb'))
    return train_x, train_y, test_x, test_y, valid_x, valid_y

def build_model(train_x, batch_size=128, dropout_rate=0.3):
    model = Sequential()
    
    # The Input Layer :
    model.add(Dense(256, kernel_initializer='normal',input_dim = train_x.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # The Hidden Layers :
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_rate))

    # The Output Layer :
    model.add(Dense(units = 1))
    model.add(Dropout(dropout_rate))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics = ['mse', 'mae'])
    print(model.summary)
    return model


def model_testing(test_x, test_y, model, history):
    pred_y = model.predict(test_x)
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

#def model_inference:
#graph = tf.get_default_graph()
    
    

@ex.config
def cfg():
    dropout_rate=0.3 
    epochs=30 
    batch_size=128

#sacred: epochs, dropout-rate, batch size, 

def model_training(train_x, train_y, valid_x, valid_y, dropout_rate, epochs, batch_size):
    model = build_model(train_x, batch_size, dropout_rate)
    history = model.fit(train_x.values, train_y.values, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(valid_x.values, valid_y.values))
    model.save('model.h5')
    return model, history

@ex.automain
def main(dropout_rate, epochs, batch_size):
    train_x, train_y, test_x, test_y, valid_x, valid_y = load_model()
    model, history = model_training(train_x, train_y, valid_x, valid_y, dropout_rate=dropout_rate, epochs=epochs, batch_size=batch_size)
    model_testing(test_x, test_y, model, history)
