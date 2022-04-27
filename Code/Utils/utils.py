# data elaboration functions
import pandas as pd
from six.moves import collections_abc
import string
import numpy as np
import re

# datetime functions
import datetime as dt

# file management functions
import os
import sys
import opendatasets as od
import pickle
from pathlib import Path

from sklearn.utils import column_or_1d

class Utils:
    def camel_to_snake(name):
        """
        Changes string from camel case to snake case
        :params: a string
        :return: a string
        """
        list_words = re.findall('([A-Z][a-z]*)', name)
        
        if len(list_words)>1:
            new_name = list_words[0].lower()
            for w in range(1, len(list_words)):
                new_name = new_name + '_' + list_words[w].lower()
        else:
            new_name = name.lower()
        return new_name
    
    def columns_camel_to_snake(df):
        """
        Changes dataframe columns from camel case to snake case
        :params: df as dataframe
        :return: a pandas dataframe
        """
        list_cols = list(df.columns)
        for name in list_cols: 
            new_name = Utils.camel_to_snake(name)
            df.rename(columns = {name: new_name}, inplace=True)
        return df
    
    def find_date(df):
            """
            Finds date columns in a dataframe
            :params: df as dataframe
            :return: a string
            """
            dates = list(df.select_dtypes(include=['datetime','datetime64[ns, UTC]']).drop_duplicates().columns)
                
            if len(dates)==1:
                print('find_date, date_col found:', dates)
                date_col = dates[0]
            elif len(dates)==0:
                dates = list(df.select_dtypes(include=['period[M]']).drop_duplicates().columns)
                print('find_date, date_col found:', dates)
                date_col = dates[0]
            else:
                date_col = dates.copy()
            
            if (len(date_col)==0):
                raise Exception('find_date, no date_col found')      
                
            return date_col
        
    def find_match_in_list(list_to_match, match_to_find):
            """
            Finds a match in a list given a list of possible words to match
            :params: list to match as a list, match_to_find as a list of words to match
            :return: a list
            """

            list_to_match = list(dict.fromkeys(list_to_match))
            match_list = list()
            for m in match_to_find:
                match_list.extend([el for el in list_to_match if isinstance(el, collections_abc.Iterable) and (m in el)])

            match_list = list(dict.fromkeys(match_list))
            return match_list
        
    def delta_format(delta: np.timedelta64) -> str:
        """
        Identifies frequency in numpy timedelta
        :params: numpy timedelta
        :return: a string
        """
        try:
            days = delta.astype("timedelta64[D]") / np.timedelta64(1, 'D')
            hours = int(delta.astype("timedelta64[h]") / np.timedelta64(1, 'h') % 24)
        except:
            days = delta / np.timedelta64(1, 'D')
            hours = int(delta / np.timedelta64(1, 'h') % 24)

        if days > 0 and hours > 0:
            return f"{days:.0f} d, {hours:.0f} h"
        elif days > 0:
            return f"{days:.0f} d"
        else:
            return f"{hours:.0f} h"
            
    def find_freq(timedelta):
        """
        Finds frequency in numpy timedelta
        :params: numpy timedelta
        :return: a string
        """
        if ('d' in timedelta):
            return 'D'
        elif ('h' in timedelta) & ('d' not in timedelta):
            return 'H'
        else:
            print('find_freq: could not infer frequency')
            
    def find_freq_in_dataframe(df, date_var):
        """
        Finds frequency in pandas dataframe
        :params: df as pandas dataframe, date_var as string
        :return: a string
        """
        freq = pd.Series(df[date_var].unique()).dt.freq
        return freq
    
    def get_project_root(Path):
        """
        Finds the parent folder of the parent folder 
        :params: Path
        :return: Path
        """
        return Path(__file__).parent.parent

    def add_daily_date(df):
        """
        Adds a date variable at daily frequency to dataframe
        :params: pandas dataframe
        :return: pandas dataframe
        """
        
        date_var = Utils.find_date(df)
        delta = abs(np.diff(df[date_var])).mean()
        timedelta = Utils.delta_format(delta)
        freq = Utils.find_freq(timedelta)
        
        # Creating date_daily 
        
        if (freq == 'H'):
            if isinstance(date_var,list)==False:
                new_var_hour_str = date_var + '_hour_str'
                new_var = date_var + '_daily'
                df.loc[:, new_var_hour_str] = df.loc[:, date_var].dt.strftime('%Y-%m-%d %H:%M:%S')
                df.loc[:, new_var] = pd.to_datetime(df.date_hour_str.apply(lambda x: x.split(' ')[0]), format = '%Y-%m-%d')
                df.drop(columns=new_var_hour_str, inplace=True)
            else:
                for d in date_var:    
                    new_var_hour_str = d + '_hour_str'
                    new_var = d + '_daily'                
                    df.loc[:, new_var_hour_str] = df.loc[:, d].dt.strftime('%Y-%m-%d %H:%M:%S')
                    df.loc[:, new_var] = pd.to_datetime(df.date_hour_str.apply(lambda x: x.split(' ')[0]), format = '%Y-%m-%d')
                    df.drop(columns=new_var_hour_str, inplace=True)
        elif (freq == 'D'):
            if (isinstance(date_var,list)==False):
                new_var = date_var + '_daily'
                if (new_var not in list(df.columns)):
                    df.rename(columns = {date_var: date_var + '_daily'}, inplace=True)
                else:
                    print('add_daily_date: data are in daily format')                
            else:
                for d in date_var:
                    new_var = d + '_daily'
                    if (new_var not in list(df.columns)):
                        df.rename(columns = {date_var: date_var + '_daily'}, inplace=True)
                    else:
                        print('add_daily_date: data are in daily format')                
        return df
    
    def find_categorical_variables(df):
        """
        Finds categorical variables in pandas dataframe
        :params: pandas dataframe
        :return: pandas dataframe
        """
        
        categorical_dtypes = ['category', 'bool']
        date_dtypes = ["datetime64[ns, UTC]"]
        list_categorical = []
        for col in list(df.columns):
            try:
                df[col] = df[col].apply(lambda x: int(x))
                if (df[col].dtype.name in categorical_dtypes) & (df[col].dtype.name not in date_dtypes):
                    list_categorical = list_categorical + [col]
                elif all(df[col].isin([0, 1])) & (df[col].dtype.name not in date_dtypes):
                    list_categorical = list_categorical + [col]
                elif (df[col].dtype.name not in date_dtypes):
                    list_categorical = list_categorical.copy()
            except:
                list_categorical = list_categorical.copy()
                
        return list_categorical
        
    def resample_data(df, id, date_var, sampling, dict_grouping):
        """
        Resample the data to a particular frequency
        :params: df as pandas dataframe, id as string, date_var as string, 
            sampling as string of frequency and dict_grouping as dictionary
        :return: a Pandas dataframe
        """
         
        wanted_keys = list(set(dict_grouping.keys()) - set([id, date_var]))
        dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in wanted_keys])
        list_variables = list(dictfilt(dict_grouping, wanted_keys).keys())      
        
        # df setup for merge    
        id_list = list(df[id].unique())
        df_resampled = df.loc[df[id] == id_list[0], [date_var, id, list_variables[0]]].drop_duplicates([date_var]).resample(sampling, on=date_var).agg({list_variables[0]: dict_grouping[list_variables[0]]}).reset_index()    
        df_resampled.loc[:, id] = id_list[0]
        print('resample_data: variable', list_variables[0])
        for i in range(1, len(id_list)):
            m = df.loc[df[id] == id_list[i], [date_var, id, list_variables[0]]].drop_duplicates([date_var]).resample(sampling, on=date_var).agg({list_variables[0]: dict_grouping[list_variables[0]]}).reset_index()    
            m.loc[:, id] = id_list[i]
            df_resampled = pd.merge(df_resampled, m, on=[id, date_var, list_variables[0]], how='outer', validate = '1:1')
        print('resample_data: variable', list_variables[0], 'completed' )
    
        # df loop for merge
        for k in range(1, len(list_variables)):
            df_m = df.loc[df[id] == id_list[0], [date_var, id, list_variables[k]]].drop_duplicates([date_var]).resample(sampling, on=date_var).agg({list_variables[k]: dict_grouping[list_variables[k]]}).reset_index()    
            df_m.loc[:, id] = id_list[0]
            print('resample_data: variable', list_variables[k])
            for i in range(1, len(id_list)):
                m = df.loc[df[id] == id_list[i], [date_var, id, list_variables[k]]].drop_duplicates([date_var]).resample(sampling, on=date_var).agg({list_variables[k]: dict_grouping[list_variables[k]]}).reset_index()    
                m.loc[:, id] = id_list[i]
                df_m = pd.merge(df_m, m, on=[id, date_var, list_variables[k]], how='outer', validate = '1:1')
            
            df_resampled = pd.merge(df_resampled, df_m, on=[id, date_var], how='outer', validate = '1:1')
            print('resample_data: variable', list_variables[k], 'completed' )
        print(df_resampled)
        return df_resampled 
    
    
    def add_seq(df, date_var, serie, freq, end_date='', start_date=''):
        """
        Creates a sequence of completes date/hours to a dataframe
        :params: dataframe in long format to add date/hour observations, date_var as string, 
            serie or id as string or list, freq as datetime.timedelta end and start date in format "%dd/%mm/%YYYY"
        :return: a Pandas dataframe
        """        
        if isinstance(serie, list)==False:
            seq = pd.DataFrame() 
            serie_list = list(df.loc[:, serie].unique())
            for i in serie_list:
                if start_date == '':
                    start_date = min(df.loc[df[serie]==i, date_var])
                else:
                    start_date = pd.to_datetime(start_date, dayfirst=True)
                    
                if end_date == '':
                    end_date = max(df.loc[df[serie]==i, date_var])
                else:
                    end_date = pd.to_datetime(end_date, dayfirst=True)
                
                # Sequence        
                time_range = pd.Series(pd.date_range(
                        start=start_date, end=end_date, freq=freq))
                
                print('Adding sequence to serie', i, 'as', 
                        serie_list.index(i) + 1, 'of', len(serie_list))
                temp = pd.DataFrame.from_dict({serie: [i] * len(time_range), 'date': time_range})
                temp.rename(columns={'date': date_var}, inplace=True)
                seq = pd.concat([seq, temp], axis=0, ignore_index=True)
            
            serie = [serie, date_var]
        else:
            seq = pd.DataFrame() 
            serie_list = df.loc[:, serie].drop_duplicates().reset_index(drop=True)
                            
            row_list = serie_list.shape[0]
            col_list = serie_list.shape[1]
            for i in range(0, row_list, 1):     
                print('Adding sequence to serie', i + 1, 'of', row_list)
                dict = {}
                for c in range(0, col_list, 1):   
                    col_name = serie_list.columns[c]
                    id_col = serie_list.loc[i,col_name]
                    if start_date == '':
                        start_date = min(df.loc[(df[col_name]==id_col), date_var])
                    else:
                        start_date = pd.to_datetime(start_date, dayfirst=True)
                        
                    if end_date == '':
                        end_date = max(df.loc[(df[col_name]==id_col), date_var])
                    else:
                        end_date = pd.to_datetime(end_date, dayfirst=True)
                    
                    # Sequence        
                    time_range = pd.Series(pd.date_range(
                            start=start_date, end=end_date, freq=freq))            
                                            
                    temp_col = {col_name: [serie_list.loc[i,col_name]]* len(time_range)}
                    dict.update(temp_col)                    
           
                temp = pd.DataFrame.from_dict(dict)
                temp.loc[:, date_var] = time_range
                seq = pd.concat([seq, temp], axis=0, ignore_index=True)            
            serie.extend([date_var])
            
        duplicates = seq.loc[:, serie].duplicated().any()
        if duplicates==True:
            raise Exception(print("add_seq: there are duplicates in sequence"))
        else:
            print("add_seq: there are NO duplicates in sequence")
        df_seq = pd.merge(seq, df, on=serie, how='left', validate='1:1')
        
        duplicates_in_df_seq = df_seq.loc[:, serie].duplicated().any()
        if duplicates_in_df_seq==True:
            raise Exception(print("add_seq: there are duplicates when adding sequence"))
        else:
            print("add_seq: there are NO duplicates when adding sequence")

        print('Total serie to forecast:', len(df_seq.loc[:, serie].drop_duplicates()))

        return df_seq
    

    

        
   
    
   