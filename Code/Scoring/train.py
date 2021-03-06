# file management functions
import os
import glob

# data elaboration functions
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import re
import pickle

# datetime functions
import datetime as dt

# AI functions
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# custom functions
from Code.Utils.utils import Utils

class Training:    
    def train(dict_model_to_train, model):
        """
        Generate train
        :params: dict_model_to_train as dictionary, model as string
        :return: a pandas dictionary
        """
        y = dict_model_to_train['y']
        X_train = dict_model_to_train['X_train']
        Y_train = dict_model_to_train['Y_train']
        list_id = dict_model_to_train['list_id']
        regressors_list = sorted(list(set(list(X_train.columns)) - set(list_id)))
        
        # Training
        print('Training')             
        
        X = X_train.loc[:, sorted(regressors_list)].copy().reset_index(drop=True)
        Y = Y_train.loc[:, y].copy().reset_index(drop=True)
    
        trained_model = model.fit(X,Y)  
        
        print('Training completed')
        return trained_model
    

        

    