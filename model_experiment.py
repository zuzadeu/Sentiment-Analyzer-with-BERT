# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:04:10 2020

@author: zuzan
"""
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import collections
import numpy as np

#def get_data

def model(dropout_rate, lstm_units, dense_units):
    model = Sequential()
    model.add(LSTM(units = 4, activation = 'sigmoid'))
    model.add(Dense(units = 1))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    print(model.summary)
    return model


def model_testing(test_y, pred_y, history):
    mae = mean_absolute_error(pred_y, test_y)
    mse = mean_squared_error(pred_y, test_y)
    print(f'MSE = {mse}')
    pyplot.plot(history.history['mean_squared_error'])
    print(f'MAE = {mae}')
    pyplot.plot(history.history['mean_absolute_error'])

#def print_history(history)


def model_training(dropout_rate, lstm_units, dense_units, train_x, train_y, epochs, batch_size):
    model = model(dropout_rate, lstm_units, dense_units)
    history = model.fit(train_x, train_y, epochs=10, batch_size=128, verbose=2)
    model.saveweigths('model.h5')
    pred_y = model.predict(test_x)
    model_testing(test_y, pred_y, history)
    return model, history


#def model_inference:
#graph = tf.get_default_graph()
.
.
.

