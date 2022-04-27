# file management functions
import os
import glob
from pyexpat.errors import XML_ERROR_UNEXPECTED_STATE

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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# custom functions
from Code.Utils.utils import Utils

class Forecasting:
    def forecast(dict_test, trained_model):
        """
        Generate forecast
        :params: dict_test as dictionary, trained_model as dictionary from training
        :return: a dictionary
        """
        X_test = dict_test['X_test']
        date_array_test = dict_test["date_array"]
        list_id = dict_test['list_id']
        date = Utils.find_date(dict_test['y_tilda'])
        
        # Regressors list
        regressors_list = sorted(list(set(list(X_test.columns)) - set(list_id)))
        
        # Forecasting    
        print('Forecasting')
         
        y_test = X_test.loc[:, regressors_list].copy()
        y_hat = trained_model.predict(y_test)
                    
        ### Adjusting negative values
        y_hat_series_pos = y_hat.copy()
        y_hat_series_pos[y_hat_series_pos < 0] = 0

        forecasted_model = {'df_fcst': pd.DataFrame({date: date_array_test, 'fcst': y_hat_series_pos})}

        print('Forecasting completed')
        return forecasted_model
    
    def call_forecasting_function(func, *args):
        """        
        Calls forecasting function       
        :params: func as function name, *args as dictionary of arguments of the function
        :return: the result of the function
        """
        from Code.Scoring.forecast import Forecasting
        func_dict = {'xgboost': Forecasting.forecast_xgboost, 'xgboost_length_ts': Forecasting.forecast_xgboost_length_ts, 'xgboost_seasons': Forecasting.forecast_xgboost_seasons}
        result = func_dict.get(func)(*args)
        return result