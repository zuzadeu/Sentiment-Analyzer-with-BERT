
import pandas as pd
import numpy as np
from bert_serving.client import BertClient

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

def remove_nonwords(df, column):
    """
    Replace non-words from the beginning and end of the string
    Parameters
    ----------
    df : data frame
    column : column name with input text
    
    Returns
    -------
    df : data frame with cleaned text
    """
    reg = r"^[^a-zA-Z]*|[^a-zA-Z]*$"
    df[column] = df[column].str.replace(reg, "")
    return df
    
def remove_empty_rows(df, column):
    df = df[df[column].astype(bool)]
    df[~df[column].isnull()].reset_index(drop=True)
    return df

def text_cleaning(df, column):
    df = remove_numbers(df, column)
    df = remove_emails(df, column)
    df = remove_html(df, column)
    df = remove_special(df, column)
    df = remove_nonwords(df, column)
    df = remove_empty_rows(df, column)
    return df

def model_inference(model, sentence):
    bc = BertClient()
    sentence = bc.encode(sentence)
    score = model.predict(sentence)
    pred_y = np.loadtxt("pred_y.csv", delimiter=",")
    min_score, max_score = min_max_mean_sentiment(pred_y)
    norm_score = (score - min_score)/(max_score - min_score)
    return norm_score

def min_max_mean_sentiment(pred_y):
    return float(min(pred_y)), float(max(pred_y))