
# data elaboration functions
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
# custom functions
from Code.Utils.utils import Utils
from Code.Scoring.kpi import Kpi

class Scoring:
    def find_best_algorithm(y, dict_train, dict_test, dict_algorithms, out_of_sample):
        """
        Finds the best performing algorithm in terms of min mean absolute error
        :params: y as string, dict_train as dictionary, dict_test as dictionary, dict_algorithm as dictionary, out_of_sample as string
        :return: a string
        """
        try:
            dict_kpi = Kpi.find_mae(y, dict_train, dict_test, dict_algorithms)
            # Best model        
            df_best_model = pd.DataFrame.from_dict(dict_kpi, orient='index').reset_index()
            df_best_model.rename(columns={'index': 'model', 0: 'mae'}, inplace=True)
            best_model = df_best_model.loc[df_best_model.mae==df_best_model.mae.min(), 'model'].reset_index(drop=True)[0]
        except:
            print('best model could not be computed, no KPI available, using out of sample algorithm. Check to have an overlap between training and test sets dates!')
            best_model = out_of_sample
        return best_model
    
    def stats_per_site(df, id, date_var):
        """
        Helper function to identify amount of data per site
        :params: df as pandas dataframe, id as string, date_var as string
        :return: a pandas dataframe
        """
        return pd.DataFrame(
            [{
                id: site, 
                "Years": df.loc[(df[id] == site), date_var].dt.year.unique(), 
                "Max Timestamp": df.loc[(df[id] == site), date_var].max(), 
                "Min Timestamp": df.loc[(df[id] == site), date_var].min(),
                "Samples": df[(df[id] == site)].count().sum()
                } for site in df[id].unique()]
        ).sort_values("Samples", ascending=False)
        
    def resample_train_data(df, date_var, id, predict_col, sampling="D"):
        """
        Resample the data to a particular frequency
        :params: df as pandas dataframe, date_var as string, id as string, sampling as string of frequency
        """
        try: 
            df_resampled = df.groupby(id) \
                    .apply(lambda group: group.set_index(date_var).resample(sampling).interpolate(method="time")) \
                    .reset_index(level=1) \
                    .reset_index(drop=True) \
                    .dropna(subset=[predict_col])
        except:
            print('resample_train_data: data are already at', sampling, 'frequency')
            df_resampled = df.copy()
                    
        return df_resampled

   
