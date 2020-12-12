# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:23:13 2020

@author: KX764QE
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import config
import preprocessor as pp

categorical_encoder = Pipeline(steps = [
    ('encoder', OneHotEncoder(categories = 'auto', drop = 'first', sparse = False, handle_unknown='error'))
    ])

numerical_scaler = Pipeline(steps = [
    ('scaler', StandardScaler())
    ])

encoder_scaler_pipe = ColumnTransformer(
    transformers = [
        ('categorical_encoder', categorical_encoder, config.CATEGORICAL_VARS_ENCODE),
        ('numerical_scaler', numerical_scaler, config.NUMERICAL_VARS_SCALE)
        ],
    remainder = 'passthrough'
    )

churn_pipe = Pipeline(
    [
     ('categorical_imputer', 
      pp.CategoricalImputer(variables = config.CATEGORICAL_VARS_WITH_NA)),
     
     ('numerial_imputer',
      pp.NumericalImputer(variables = config.NUMERICAL_VARS_WITH_NA)),
     
     ('encoder_scaler', 
      encoder_scaler_pipe),
     
     ('model', LogisticRegression(C = 0.01, random_state= config.RANDOM_SEED))
     ])