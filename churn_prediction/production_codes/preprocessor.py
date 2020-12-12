# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:23:26 2020

@author: KX764QE
"""

from sklearn.base import BaseEstimator, TransformerMixin

import config

# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(lambda x: 'missing' if x in config.MISSING_VALUES else x)
        
        return X

# numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y = None):
        # persist median in a dictory
        self.imputer_dict = {}
        
        for feature in self.variables:
            self.imputer_dict[feature] = X[feature].median()
        return self

    def transform(self, X, y = None):
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict[feature], inplace = True)
    
        return X

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    