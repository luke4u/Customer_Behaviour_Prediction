# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:22:59 2020

@author: KX764QE
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import config 

def read_raw_data(path_to_file):
    df = pd.read_csv(path_to_file, usecols = config.FEATURES + config.TARGET)
    return df

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df[config.FEATURES], 
                                                        df[config.TARGET],
                                                        test_size = 0.2,
                                                        random_state = config.RANDOM_SEED)
    
    X_train.reset_index(drop = True, inplace = True)
    X_test.reset_index(drop = True, inplace = True)
    
    y_train.reset_index(drop = True, inplace = True)
    y_test.reset_index(drop = True, inplace = True)
    
    return X_train, X_test, y_train, y_test

def complete_data_analys(df, var):
    df = df.dropna(axis = 0, how = 'any', thresh = 1, subset = var)
    return df

if __name__ == '__main__':
    """
    read raw data
    drop rows with only na based on age column
    split data into train and test sets
    """
    df = read_raw_data(config.RAW_DATA_FILE)
    df = complete_data_analys(df, ['age'] )
    X_train, X_test, y_train, y_test = split_data(df)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    
    
    
    