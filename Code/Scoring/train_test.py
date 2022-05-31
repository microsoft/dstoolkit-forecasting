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
    def define_train_test_set_dates(df, y, train_start_date, train_end_date, test_start_date, test_end_date, test_size=0.33):
        """        
        Defines train and test dates if left blank      
        :params: df as pandas dataframe, y as string, train_start_date as string in format '%Y-%m-%d', train_end_date as string in format '%Y-%m-%d', test_start_date as string in format '%Y-%m-%d', test_end_date as string in format '%Y-%m-%d', test_size as percentage
        :return: a dictionary 
        """
        date_var = Utils.find_date(df)
        min_train_start_date = df.loc[(df[y].isnull()==False), date_var].min()
        max_train_end_date = df.loc[(df[y].isnull()==False), date_var].max()
        min_test_start_date = df.loc[(df[y].isnull()==True), date_var].min()
        max_test_end_date = df.loc[(df[y].isnull()==True), date_var].max()
        range = pd.date_range(start=min_train_start_date,end=max_train_end_date)   
        
        # Test set: identify latest date and set test set as latest date - test size offset
        if test_end_date=='':            
            test_end_date =  max_test_end_date
        else:
            test_end_date = pd.to_datetime(test_end_date, format='%Y-%m-%d')
            
        if (test_start_date=='') & (min_test_start_date>max_train_end_date):
            offset_date = pd.to_datetime(test_end_date, format='%Y-%m-%d') - pd.DateOffset(n = round(len(range)*test_size, 0) )
            test_start_date = min([offset_date, min_test_start_date])
            #test_start_date = (pd.to_datetime(test_end_date, format='%Y-%m-%d') - offset_date).strftime('%Y-%m-%d')
        elif (test_start_date==''):
            test_start_date = min_train_start_date
        else:
            test_start_date = pd.to_datetime(test_start_date, format='%Y-%m-%d')
            
        # Train set: set train set from test start date -1 to test to minimum date available
        if train_start_date=='':   
            train_start_date = min_train_start_date
        else:
            train_start_date = pd.to_datetime(train_start_date, format='%Y-%m-%d')
            
        if train_end_date=='':
            train_end_date = max_train_end_date
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
        elif (train_start_date != '') & (isinstance(train_start_date, str)):
            train_start_date = pd.to_datetime(train_start_date, dayfirst=True)
        else:
            print('Train start date is already a date')
            
        print('Train start date is', train_start_date)

        if train_end_date == '':
            train_end_date = max(df.loc[df[y].notnull(), date_var])
        elif (train_end_date != '') & (isinstance(train_end_date, str)):
            train_end_date = pd.to_datetime(train_end_date, dayfirst=True)
        else:
            print('Train end date is already a date')            
            
        print('Train end date is', train_end_date)

        ### Slicing by observation
        df_sliced = df.loc[(~df.loc[:, y].isnull()) & (df.loc[:, date_var]>=train_start_date) & (df.loc[:, date_var]<=train_end_date), ].reset_index(drop=True)
        print('Train shape before removing nan is', df_sliced.shape[0])
        
        # Removing additional nan
        train = df_sliced[df_sliced.isnull()==False].sort_values(by=date_var).reset_index(drop=True)
        train_start_date = min(df_sliced.loc[:, date_var])
        print('Min date AFTER removing nan is', train_start_date)
        train_end_date = max(df_sliced.loc[:, date_var])
        print('Max date AFTER removing nan is', train_end_date)
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
        historical_data = df.loc[df[date_var]>=min(df.loc[df[y].notnull(), date_var]), [date_var, y]].reset_index(drop=True)
        
        ### Create final dict
        dict_train = {'X_train': X_train, 'Y_train': Y_train, 'date_array': date_array, 'y': y, 'list_id': list_id, 'train_start_date': train_start_date, 'train_end_date': train_end_date, 'historical_data': historical_data}
        
        return dict_train

    def def_test(df, y, list_id, test_start_date='', test_end_date=''):
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
            test_end_date = df.loc[(df[y].isnull()==True), date_var].max()
        else:
            test_end_date = pd.to_datetime(test_end_date, dayfirst=True)
        print('Test end date is', test_end_date)

        ### Slicing by observation
        df_sliced = df.loc[(df[date_var]>= test_start_date) & (df[date_var] <= test_end_date), ].reset_index(drop=True)
        test = df_sliced.sort_values(by=date_var)
        test_start_date = min(df_sliced.loc[:, date_var])
        test_end_date = max(df_sliced.loc[:, date_var])
        
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

        dict_test = {'X_test': X_test, 'y_tilda' : y_tilda, 'date_array': date_array, 'y': y, 'list_id': list_id, 'test_start_date': test_start_date, 'test_end_date': test_end_date, 'historical_data': historical_data}

        return dict_test

    