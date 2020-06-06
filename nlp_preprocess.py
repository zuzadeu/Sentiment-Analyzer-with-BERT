# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:48:03 2020

@author: zuzan
"""
import pandas as pd
import numpy as np
import os
from bert_serving.client import BertClient
from imblearn.over_sampling import SMOTE
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import smogn
import seaborn

test_size = 0.1
valid_size = 0.2
random_state = 42

np.random.seed(1)

def remove_numbers(df, column):
    """
    Replace digits with "numtoken"
    Parameters
    ----------
    df : data frame
    column : column name with input text
    
    Returns
    -------
    df : data frame with cleaned text
    """
    reg = r"[0-9][0-9]*"
    token = "numtoken"
    df[column] = df[column].str.replace(reg, token)
    return df
    
    
def remove_emails(df, column):
    """
    Replace emails with "emtoken"
    Parameters
    ----------
    df : data frame
    column : column name with input text
    
    Returns
    -------
    df : data frame with cleaned text
    """
    reg = r"[A-Za-z.]*@[a-z]*\-*[a-z]*.com|[A-Za-z\-]*.com"
    token = "emtoken"
    df[column] = df[column].str.replace(reg, token)
    return df
        
    
def remove_html(df, column):
    """
    Replace html code with "htmltoken"
    Parameters
    ----------
    df : data frame
    column : column name with input text
    
    Returns
    -------
    df : data frame with cleaned text
    """
    reg = r"<[/]?(((http|ftp)[s]?://)?www.|((http|ftp)[s]?://)|mailto:|ro_)(?:[a-zA-Z]|[0-9]|[$-=?-_@.&+]|[!*\(\),]|\
        (?:%[0-9a-fA-F][0-9a-fA-F]))+"
    token = "htmltoken"
    df[column] = df[column].str.replace(reg, token)
    reg = "[/]?(((http|ftp)[s]?://)?www.|((http|ftp)[s]?://)|mailto:|ro_)(?:[a-zA-Z]|[0-9]|\
        [$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    df[column] = df[column].str.replace(reg, token)
    return df
        
    
def remove_special(df, column):
    """
    Replace lines and tabs with space
    Parameters
    ----------
    df : data frame
    column : column name with input text
    
    Returns
    -------
    df : data frame with cleaned text
    """
    reg = r"\n|\t"
    space = " "
    df[column] = df[column].str.replace(reg, space)
    return df
    
    
def train_valid_test_split(df):
    y = df['airline_sentiment_confidence']
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
    bc = BertClient()
    train_seq_x = bc.encode(list(train_x.values))
    test_seq_x = bc.encode(list(test_x.values))
    valid_seq_x = bc.encode(list(valid_x.values))
    
    train_x = pd.DataFrame(data=train_seq_x, index=train_x.index)
    test_x = pd.DataFrame(data=test_seq_x, index=test_x.index)
    valid_x = pd.DataFrame(data=valid_seq_x, index=valid_x.index)
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
    df = pd.read_csv('Tweets.csv', encoding='utf-8')
    df = remove_numbers(df, 'text')
    df = remove_emails(df, 'text')
    df = remove_html(df, 'text')
    df = remove_special(df, 'text')
    print('Train, test, valid split...')
    train_x, train_y, valid_x, valid_y, test_x, test_y = train_valid_test_split(df)
    #print(train_x.head(5))
    #print(train_x.columns)
    max_seq_len = max_sequence_length(train_x)
    print(f'Majority of sentences contain {max_seq_len} words.')
    if os.path.exists('bert_train_x.pkl') == False:
        print('Bert encoding...')
        train_x, test_x, valid_x = BERT_embeddings(train_x, test_x, valid_x, valid_y, max_seq_len)
        train_x.to_pickle('bert_train_x.pkl')
        test_x.to_pickle('bert_test_x.pkl') 
        valid_x.to_pickle('bert_valid_x.pkl')
    else:
        print('Bert embeddings loading...')
        train_x = pd.read_pickle('bert_train_x.pkl')
        test_x = pd.read_pickle('bert_test_x.pkl') 
        valid_x = pd.read_pickle('bert_valid_x.pkl')
    print('Padding...')
    #train_x, valid_x = padding(train_x, valid_x, max_seq_len)
    train_x, train_y, valid_x, valid_y = oversample_train_valid(train_x, train_y, valid_x, valid_y)
    save_dataframes(train_x, train_y, test_x, test_y, valid_x, valid_y)
    
__main__()