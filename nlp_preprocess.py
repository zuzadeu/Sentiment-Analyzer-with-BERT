# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:48:03 2020

@author: zuzan
"""
import pandas as pd
import numpy as np
import os
from bert_serving.client import BertClient
from sklearn.model_selection import train_test_split
import smogn
import seaborn
from common import *
import logging

logging.basicConfig(level=logging.ERROR)

test_size = 0.1
valid_size = 0.2
random_state = 42

np.random.seed(1)

def train_valid_test_split(df):
    y = df['value']
    x = df['text']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, 
                                                        random_state=random_state)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, 
                                                          test_size=valid_size, 
                                                          random_state=random_state)
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y
  
def max_sequence_length(train_x):
    return int(train_x.str.split().str.len().quantile(0.9))


def BERT_embeddings(train_x, test_x, valid_x, valid_y, max_seq_len):
    bc = BertClient(ip='192.168.1.43')
    train_seq_x = bc.encode(list(train_x.values))
    test_seq_x = bc.encode(list(test_x.values))
    valid_seq_x = bc.encode(list(valid_x.values))
    
    train_x = pd.DataFrame(data=train_seq_x, index=train_x.index)
    test_x = pd.DataFrame(data=test_seq_x, index=test_x.index)
    valid_x = pd.DataFrame(data=valid_seq_x, index=valid_x.index)
    return train_x, test_x, valid_x


def save_dataframes(train_x, train_y, test_x, test_y, valid_x, valid_y):
    train_x.to_pickle('train_x.pkl')
    test_x.to_pickle('test_x.pkl') 
    valid_x.to_pickle('valid_x.pkl')
    train_y.to_pickle('train_y.pkl')
    test_y.to_pickle('test_y.pkl') 
    valid_y.to_pickle('valid_y.pkl')
    
    
def __main__():
    if os.path.exists('train_x.pkl') == False:
        print('Text cleaning...')
        sentences = pd.read_csv('dictionary.txt', encoding='utf-8', delimiter = '|', names = ['text', 'id'])
        labels = pd.read_csv('sentiment_labels.txt', encoding='utf-8', delimiter = '|', names = ['id', 'value'])
        df = sentences.merge(labels, on='id', how='left')
        df = text_cleaning(df, 'text')
        print('Train, test, valid split...')
        train_x, train_y, valid_x, valid_y, test_x, test_y = train_valid_test_split(df)
        max_seq_len = max_sequence_length(train_x)
        print(f'Majority of sentences contain {max_seq_len} words.')
        print('Bert encoding...')
        train_x, test_x, valid_x = BERT_embeddings(train_x, test_x, valid_x, valid_y, max_seq_len)
        save_dataframes(train_x, train_y, test_x, test_y, valid_x, valid_y)
        print('Saved!')
    else:
        print('Bert embeddings exist.')
    

__main__()