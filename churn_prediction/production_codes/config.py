# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:09:34 2020

@author: KX764QE
"""
import numpy as np


# data
RAW_DATA_FILE = "app_churn_data.csv"
PIPELINE_NAME = 'churn_pipeline'

TARGET = ['churn']

# input variables
FEATURES = ['age', 'housing', 'credit_score', 'deposits', 'withdrawal',
       'purchases_partners', 'purchases', 'cc_taken', 'cc_recommended',
       'cc_disliked', 'cc_liked', 'cc_application_begin', 'app_downloaded',
       'web_user', 'app_web_user', 'ios_user', 
       'registered_phones', 'payment_type', 'waiting_4_loan', 'cancelled_loan',
       'received_loan', 'rejected_loan', 'zodiac_sign',
       'left_for_two_month_plus', 'left_for_one_month', 'rewards_earned',
       'reward_rate', 'is_referred']

# numerical variables with na
NUMERICAL_VARS_WITH_NA = ['age', 'credit_score', 'rewards_earned']

# categorical variables with na
CATEGORICAL_VARS_WITH_NA = ['housing', 'payment_type', 'zodiac_sign']

# categorical variables to encode
CATEGORICAL_VARS_ENCODE = ['housing', 'payment_type', 'zodiac_sign']

# numerical variables to scale
NUMERICAL_VARS_SCALE = ['age', 'credit_score', 'deposits', 'withdrawal', 'purchases_partners', 
                        'purchases',  'cc_taken',  'cc_recommended', 'cc_disliked',  'cc_liked', 
                        'cc_application_begin',  'app_downloaded', 'web_user',  'app_web_user',
                        'ios_user',  'registered_phones', 'waiting_4_loan',  'cancelled_loan',  
                        'received_loan', 'rejected_loan', 'left_for_two_month_plus', 'left_for_one_month',
                        'rewards_earned',  'reward_rate',  'is_referred']

# missing values
MISSING_VALUES = ['NA', 'Na', 'na', np.nan ]

# random seed
RANDOM_SEED= 101