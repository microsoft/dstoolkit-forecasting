
# data elaboration functions
from attr import validate
import pandas as pd
from six.moves import collections_abc
import string
import numpy as np

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
        dict_kpi = {}
        
        # Training and forecasting
        for m in list(dict_models.keys()):  
            print('kpi for model', m)
            model = dict_models[m]       
            trained_model = Training.train(dict_train, model)
            forecasted_model = Forecasting.forecast(dict_test, trained_model = trained_model)
            y_tilda = dict_test['y_tilda'].copy()
            y_tilda_date = Utils.find_date(y_tilda)
            y_hat = forecasted_model['df_fcst'].copy()
            y_hat_date = Utils.find_date(y_hat)
            
            df_merge = pd.merge(y_tilda, y_hat, left_on=y_tilda_date, right_on=y_hat_date, how='inner', validate='1:1')
            mae = mean_absolute_error(df_merge[y], df_merge['fcst'])
            dict_kpi[m] = mae

        return dict_kpi