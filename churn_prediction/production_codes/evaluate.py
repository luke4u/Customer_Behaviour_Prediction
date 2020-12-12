# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:47:45 2020

@author: KX764QE
"""

import joblib
import config
import data_management as dm

def make_prediction(input_data):
    
    _pipe_churn = joblib.load(filename = config.PIPELINE_NAME)
    
    results = _pipe_churn.predict(input_data)
    
    return results

if __name__ == '__main__':
    """
    test pipeline
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
     
    df = dm.read_raw_data(config.RAW_DATA_FILE)
    
    X_train, X_test, y_train, y_test = dm.split_data(df)
    
    pred = make_prediction(X_test)
    
    print(accuracy_score(y_test, pred))
    print(precision_score(y_test, pred))
    print(recall_score(y_test, pred))
    print(f1_score(y_test, pred))