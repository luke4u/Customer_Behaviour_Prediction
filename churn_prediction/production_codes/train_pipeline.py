# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:33:32 2020

@author: KX764QE
"""
import config
import data_management as dm
import pipeline 

import joblib

def run_training():
    """
    Train the model
    """
    
    df = dm.read_raw_data(config.RAW_DATA_FILE)
    
    X_train, X_test, y_train, y_test = dm.split_data(df)
    # print(y_train.shape)
    y_train = y_train.values.reshape(-1,)
    # print(y_train.shape)
    pipeline.churn_pipe.fit(X_train, y_train)
    
    
    joblib.dump(pipeline.churn_pipe, config.PIPELINE_NAME)
    
    
if __name__ == '__main__':
    """
    train pipeline
    """
    run_training()