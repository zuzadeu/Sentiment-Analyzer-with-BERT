# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:48:03 2020

@author: zuzan
"""
import pandas as pd
import numpy as np
import os
from bert_serving.client import BertClient
from transformers import MobileBertTokenizer
from imblearn.over_sampling import SMOTE
from keras.preprocessing.sequence import pad_sequences
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


def BERT_embeddings2(train_x, test_x, valid_x, valid_y, max_seq_len):
    tokenizer = MobileBertTokenizer.from_pretrained("bert-base-uncased")

    train_seq_x = tokenizer.batch_encode_plus(train_x.values, max_length=max_seq_len, pad_to_max_length=True, truncation=True)['input_ids']
    test_seq_x = tokenizer.batch_encode_plus(test_x.values, max_length=max_seq_len, pad_to_max_length=True, truncation=True)['input_ids']
    valid_seq_x = tokenizer.batch_encode_plus(valid_x.values, max_length=max_seq_len, pad_to_max_length=True, truncation=True)['input_ids']
    
    train_x = pd.DataFrame(data=train_seq_x)#, index=train_x.index)
    test_x = pd.DataFrame(data=test_seq_x)#, index=test_x.index)
    valid_x = pd.DataFrame(data=valid_seq_x)#, index=valid_x.index)
    return train_x, test_x, valid_x

    
def padding(train_x, valid_x, max_seq_len):
    train_x = pad_sequences(train_x, maxlen=max_seq_len, padding='pre')
    valid_x = pad_sequences(valid_x, maxlen=max_seq_len, padding='pre')
    return train_x, valid_x
    
#Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise
def apply_SMOGN(x, y):
    data = pd.concat([x, y], axis=1, 
                     ignore_index=True)
    y_col_number = data.shape[1]-1
    data = data.rename(columns = {y_col_number: 'airline_sentiment_confidence'}).reset_index()
    df_smogn = smogn.smoter(
    
    # main arguments
    data = data,           ## pandas dataframe
    y = 'airline_sentiment_confidence',          ## string ('header name')
    k = 9,                    ## positive integer (k < n)
    samp_method = 'extreme',  ## string ('balance' or 'extreme')

    # phi relevance arguments
    rel_thres = 0.5,         ## positive real number (0 < R < 1) #0.5
    rel_method = 'auto',      ## string ('auto' or 'manual')
    rel_xtrm_type = 'both',   ## string ('low' or 'both' or 'high')
    rel_coef = 1#2.25           ## positive real number (0 < R) #1.5
)
    
    x_oversampled = df_smogn['text']
    y_oversampled = df_smogn['airline_sentiment_confidence']
    
    ## plot y distribution 
    seaborn.kdeplot(y, label = "Original")
    seaborn.kdeplot(y_oversampled, label = "Modified")
    return x_oversampled, y_oversampled 
    

def oversample_train_valid(train_x, train_y, valid_x, valid_y):
    train_x, train_y = apply_SMOGN(train_x, train_y) 
    valid_x, valid_y = apply_SMOGN(valid_x, valid_y) 
    return train_x, train_y, valid_x, valid_y
        

def save_dataframes(train_x, train_y, test_x, test_y, valid_x, valid_y):
    train_x.to_pickle('train_x.pkl')
    test_x.to_pickle('test_x.pkl') 
    valid_x.to_pickle('valid_x.pkl')
    train_y.to_pickle('train_y.pkl')
    test_y.to_pickle('test_y.pkl') 
    valid_y.to_pickle('valid_y.pkl')
    
    
def __main__():
    print('Text cleaning...')
    sentences = pd.read_csv('dictionary.txt', encoding='utf-8', delimiter = '|', names = ['text', 'id'])
    labels = pd.read_csv('sentiment_labels.txt', encoding='utf-8', delimiter = '|', names = ['id', 'value'])
    df = sentences.merge(labels, on='id', how='left')
    df = text_cleaning(df, 'text')
    print('Train, test, valid split...')
    train_x, train_y, valid_x, valid_y, test_x, test_y = train_valid_test_split(df)
    #print(train_x.head(5))
    #print(train_x.columns)
    max_seq_len = max_sequence_length(train_x)
    print(f'Majority of sentences contain {max_seq_len} words.')

    # if os.path.exists('bert_train_x.pkl') == False:
    #     print('Bert encoding...')
    #     train_x, test_x, valid_x = BERT_embeddings(train_x, test_x, valid_x, valid_y, max_seq_len)
    #     train_x.to_pickle('bert_train_x.pkl')
    #     test_x.to_pickle('bert_test_x.pkl') 
    #     valid_x.to_pickle('bert_valid_x.pkl')
    # else:
    #     print('Bert embeddings loading...')
    #     train_x = pd.read_pickle('bert_train_x.pkl')
    #     test_x = pd.read_pickle('bert_test_x.pkl') 
    #     valid_x = pd.read_pickle('bert_valid_x.pkl')
    # #print('Padding...')
    # #train_x, valid_x = padding(train_x, valid_x, max_seq_len)
    # #train_x, train_y, valid_x, valid_y = oversample_train_valid(train_x, train_y, valid_x, valid_y)
    # print('Save...')
    # save_dataframes(train_x, train_y, test_x, test_y, valid_x, valid_y)

    if os.path.exists('train_x.pkl') == False:
        print('Bert encoding...')
        train_x, test_x, valid_x = BERT_embeddings2(train_x, test_x, valid_x, valid_y, max_seq_len)
        save_dataframes(train_x, train_y, test_x, test_y, valid_x, valid_y)
    else:
        print('Bert embeddings loading...')
    print('Saved!')

__main__()