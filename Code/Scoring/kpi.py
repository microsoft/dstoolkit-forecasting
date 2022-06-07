
# data elaboration functions
from attr import validate
import pandas as pd
from six.moves import collections_abc
import string
import numpy as np
import math

# datetime functions
import datetime as dt

# file management functions
import os
import sys
import opendatasets as od
import pickle
from pathlib import Path

# data science functions
from sklearn.metrics import mean_absolute_error

# custom functions
from Code.Utils.utils import Utils
from Code.Scoring.train import Training
from Code.Scoring.forecast import Forecasting

class Kpi:
    def find_mae(y, dict_train, dict_test, dict_models):
        """
        Compute mean absolute error
        :params: y as string, dict_train as dictionary, dict_test as dictionary, dict_models as dictionary
        :return: a dictionary
        """
        
        dict_test_no_nan = dict_test.copy()
        dict_test_no_nan['X_test'] = dict_test['X_test'].dropna()
        dict_test_no_nan['y_tilda'] = dict_test['y_tilda'].dropna()
        
        date_var_y_tilda = Utils.find_date(dict_test_no_nan['y_tilda'])
        dict_test_no_nan['date_array'] = dict_test_no_nan['y_tilda'].loc[:, date_var_y_tilda]
        
        # Training and forecasting
        dict_kpi = {}
        for m in list(dict_models.keys()):  
            print('kpi for model', m)
            try:
                model = dict_models[m]       
                trained_model = Training.train(dict_train, model)
                forecasted_model = Forecasting.forecast(dict_test, trained_model = trained_model)
                y_tilda = dict_test['y_tilda'].copy()
                y_tilda_date = Utils.find_date(y_tilda)
                y_hat = forecasted_model['df_fcst'].copy()
                y_hat_date = Utils.find_date(y_hat)
                
                df_merge = pd.merge(y_tilda, y_hat, left_on=y_tilda_date, right_on=y_hat_date, how='inner', validate='1:1').dropna()
                mae = mean_absolute_error(df_merge[y], df_merge['fcst'])
                dict_kpi[m] = mae
            except:
                print('kpi for model', m, 'could not be computed')

        return dict_kpi
    
    def compute_error(df, fcst, y):
        """       
        Compute error as forecast-actual
        :params: df as pandas dataframe, fcst as string as the name of the forecast columns, y as string as the name of the actual columns,
        :return: a dataframe
        """
        if 'error' in df.columns:
            df = df.drop(columns='error')
            
        df.loc[:, 'error'] = (df[fcst] - df[y])
        return df
    
    def compute_absolute_error(df, fcst, y):
        """       
        Compute absolute error as abs(forecast-actual)
        :params: df as pandas dataframe, fcst as string as the name of the forecast columns, y as string as the name of the actual columns,
        :return: a dataframe
        """
        if 'absolute_error' in df.columns:
            df = df.drop(columns='absolute_error')
            
        df.loc[:, 'absolute_error'] = abs(df[fcst] - df[y])
        return df
    
    def compute_absolute_percentage_error(df, fcst, y):
        """       
        Compute absolute % error
        :params: df as pandas dataframe, fcst as string as the name of the forecast columns, y as string as the name of the actual columns,
        :return: a dataframe
        """
        if 'absolute_error' in df.columns:
            df = df.drop(columns='absolute_error')
        
        if 'absolute_percentage_error' in df.columns:
            df = df.drop(columns='absolute_percentage_error')
        
        df = Kpi.compute_absolute_error(df, fcst, y)
        df.loc[:, 'absolute_percentage_error'] = df.loc[:, 'absolute_error']/df.loc[:, y]
        return df
    
    def compute_mean_error(df, fcst, y):
        """       
        Compute mean  error
        :params: df as pandas dataframe, fcst as string as the name of the forecast columns, y as string as the name of the actual columns,
        :return: a scalar
        """
        df = Kpi.compute_error(df, fcst, y)
        mean_error = df.loc[:, 'error'].mean()
        return mean_error
    
    def compute_mae(df, fcst, y):
        """       
        Compute mean absolute error
        :params: df as pandas dataframe, fcst as string as the name of the forecast columns, y as string as the name of the actual columns,
        :return: a scalar
        """
        df = Kpi.compute_absolute_error(df, fcst, y)
        var = 'absolute_error'
        mask = (df[var].isnull()==False) & (np.isneginf(df[var])==False) & (np.isposinf(df[var])==False)
        mae = df.loc[mask==True, var].mean()
        return mae
    
    def compute_mape(df, fcst, y):
        """       
        Compute mean absolute % error
        :params: df as pandas dataframe, fcst as string as the name of the forecast columns, y as string as the name of the actual columns,
        :return: a scalar
        """
        df = Kpi.compute_absolute_percentage_error(df, fcst, y)
        var = 'absolute_percentage_error'
        mask = (df[var].isnull()==False) & (np.isneginf(df[var])==False) & (np.isposinf(df[var])==False)
        mape = df.loc[mask==True, var].mean()
        return mape
        