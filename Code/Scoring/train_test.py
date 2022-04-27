# file management functions
import os
import glob

# data elaboration functions
import numpy as np
import pandas as pd
from openpyxl import load_workbook

# datetime functions
import datetime as dt

# custom functions
from Code.Utils.utils import Utils


class TrainTest:
    def define_train_test_set_dates(df, train_start_date, train_end_date, test_start_date, test_end_date):
        """        
        Defines train and test dates if left blank      
        :params: df as pandas dataframe, train_start_date as string in format '%Y-%m-%d', train_end_date as string in format '%Y-%m-%d', test_start_date as string in format '%Y-%m-%d', test_end_date as string in format '%Y-%m-%d'
        :return: a dictionary 
        """
        date_var = Utils.find_date(df)
        
        # Test set: identify latest date and set test set as latest date - 365 days
        if test_end_date=='':            
            test_end_date =  df.loc[:, date_var].max().strftime('%Y-%m-%d')
        else:
            test_end_date = pd.to_datetime(test_end_date, format='%Y-%m-%d')
            
        if test_start_date=='':
            test_start_date = (pd.to_datetime(test_end_date, format='%Y-%m-%d') - pd.DateOffset(days=365)).strftime('%Y-%m-%d')
        else:
            test_start_date = pd.to_datetime(test_start_date, format='%Y-%m-%d')
            
        # Train set: set train set from test start date -1 to minimum date available
        if train_start_date=='':   
            train_start_date = df.loc[:, date_var].min().strftime('%Y-%m-%d')
        else:
            train_start_date = pd.to_datetime(train_start_date, format='%Y-%m-%d')
            
        if train_end_date=='':
            train_end_date = (pd.to_datetime(test_start_date, format='%Y-%m-%d') - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        else:
            train_end_date = pd.to_datetime(train_end_date, format='%Y-%m-%d')
            
        dict_train_test_set = {'train_start_date': train_start_date, 'train_end_date': train_end_date, 'test_start_date':test_start_date, 'test_end_date': test_end_date}
        return dict_train_test_set
    
    def def_train(df, y, list_id, train_start_date='', train_end_date=''):
        """
        Define train dataset 
        :params: dataset as dataframe, y as string, list_id as list, train_start_date as string, train_end_date as string
        :return: a Pandas dataframe
        """
        date_var = Utils.find_date(df)
        df.loc[:, date_var] = df.loc[:, date_var].apply(lambda x: pd.to_datetime(dt.datetime.strftime(x, '%Y-%m-%d'), dayfirst=True))

        if train_start_date == '':
            train_start_date = min(df.loc[df[y].notnull(), date_var])
        else:
            train_start_date = pd.to_datetime(train_start_date, dayfirst=True)
        print('Train start date is', train_start_date)

        if train_end_date == '':
            train_end_date = max(df.loc[df[y].notnull(), date_var])
        else:
            train_end_date = pd.to_datetime(train_end_date, dayfirst=True)
        print('Train end date is', train_end_date)

        ### Slicing by observation
        df_sliced = df.loc[(df.loc[:, date_var]>=train_start_date) & (df.loc[:, date_var]<=train_end_date), ].reset_index(drop=True)
        print('Train shape before removing nan is', df_sliced.shape[0])
        
        # Removing additional nan
        train = df_sliced[df_sliced.isnull()==False].sort_values(by=date_var).reset_index(drop=True)
        print('Min date AFTER removing nan is', min(df_sliced.loc[:, date_var]))
        print('Max date AFTER removing nan is', max(df_sliced.loc[:, date_var]))
        print('Shape AFTER removing nan is', df_sliced.shape[0])
        
        ### Slicing by feature
        # Features set
        train_features = sorted(list(set(list(train.columns)) - set(list_id + [y])))
        y_plus_train_features = [y] + train_features
        
        # X_train and Y_train
        X_train = train.loc[:, train_features].reset_index(drop=True)
        Y_train = train.loc[:, y_plus_train_features].reset_index(drop=True)
        
        # Date array
        date_array = train.loc[:, date_var].reset_index(drop=True)
        
        # Historical data
        historical_data = df.loc[:, [date_var, y]].reset_index(drop=True)
        
        ### Create final dict
        dict_train = {'X_train': X_train, 'Y_train': Y_train, 'date_array': date_array, 'y': y, 'list_id': list_id, 'train_start_date': train_start_date, 'train_end_date': train_end_date, 'historical_data': historical_data}
        
        return dict_train

    def def_test(df, y, list_id, test_start_date='', test_end_date='', forecast_scope = 730):
        """
        Define test dataset
        :params: dataset as dataframe, y as string, list_id as list, test_start_date as string, test_end_date as string
        :return: a Pandas dictionary
        """
        date_var = Utils.find_date(df)
        df.loc[:, date_var] = df.loc[:, date_var].apply(lambda x: pd.to_datetime(dt.datetime.strftime(x, '%Y-%m-%d'), dayfirst=True))
        
        if test_start_date == '':
            test_start_date = min(df.loc[df[y].isnull()==False, date_var]) + dt.timedelta(1)
        else:
            test_start_date = pd.to_datetime(test_start_date, dayfirst=True)
        print('Test start date is', test_start_date)

        if test_end_date == '':
            test_end_date =dt.datetime.today() + dt.timedelta(forecast_scope)
        else:
            test_end_date = pd.to_datetime(test_end_date, dayfirst=True)
        print('Test end date is', test_end_date)

        ### Slicing by observation
        df_sliced = df.loc[(df[date_var]>= test_start_date) & (df[date_var] <= test_end_date), ].reset_index(drop=True)
        test = df_sliced.sort_values(by=date_var)
        
        ### Slicing by feature
        # Features set
        test_features = sorted(list(set(list(test.columns)) - set(list_id + [y])))
        y_plus_date = [date_var] + [y]
        
        # X_train, y_tilda
        X_test = test.loc[:, test_features].copy().reset_index(drop=True)
        y_tilda = test.loc[:, y_plus_date].copy().reset_index(drop=True)
        
        # Date array
        date_array = test.loc[:, date_var].copy().reset_index(drop=True)    
        
        # Historical data
        historical_data = df.loc[:, [date_var, y]].reset_index(drop=True)   

        dict_test = {'X_test': X_test, 'y_tilda' : y_tilda, 'date_array': date_array, 'y': y, 'list_id': list_id, 'test_start_date': test_start_date, 'test_end_date': test_end_date, 'forecast_scope': forecast_scope, 'historical_data': historical_data}

        return dict_test

    