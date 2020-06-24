# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 22:51:06 2020

@author: zuzan
"""
import tensorflow as tf 
from bert_serving.client import BertClient
from keras.preprocessing.sequence import pad_sequences

def model_inference(model, sentence):
    bc = BertClient()
    bc.encode(sentence)
    score = model.predict(sentence)
    print(f'sentiment score: {score}')
    

model = tf.keras.models.load_model('model.h5')
#sentence = print(input('sentence: '))
sentence = ['this is beautiful']
model_inference(model, sentence)