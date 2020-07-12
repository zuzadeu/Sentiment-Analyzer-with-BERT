# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 22:51:06 2020

@author: zuzan
"""
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from common import text_cleaning, model_inference
import pandas as pd
import numpy as np
import pickle

    
sentence = print(input('sentence: '))
df = pd.DataFrame(columns=['text']).append({'text': sentence}, ignore_index=True)
df = text_cleaning(df, 'text')
preproc_text = df.iloc[0][0]
model = tf.keras.models.load_model('model.h5')

print(sentence + ': ')
score = model_inference(model, [preproc_text])
print(f'\n {float(score)}')