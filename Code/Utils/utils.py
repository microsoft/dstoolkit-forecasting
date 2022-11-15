# data elaboration functions
import pandas as pd
import string
import numpy as np
import re
from functools import reduce
from pandasql import sqldf

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
    
    def create_folder_tree(folder_name):                
        try:
            os.makedirs(os.path.join(folder_name))
        except OSError:
            print("Creation of the directory failed or already present", folder_name)
        else:
            print("Successfully created the directory", folder_name)
        return

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
            sampling as string of frequency and dict_grouping as dictionary as {variable_to_resample: 'function_to_apply'}
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
    
    def resample_data_pandassql(df_name, id_column, date_column, freq, aggregation_per_col):
        """
        Resample the data to a particular frequency
        :params: df_name as string name of a pandas dataframe, id as string, date_var as string, 
            the sampling as string freq (e.g. 3-m, 5-h, 1-D) and aggregation_per_col as dictionary as {variable_to_resample: 'function_to_apply'}
        :return: a Pandas dataframe
        """ 
        # TO-DO: check for interval of original series
        pysqldf = lambda q: sqldf(q, globals())

        num = freq.split('-')[0]
        window = freq.split('-')[1]


        for i in set(aggregation_per_col.values()):
                if i.upper() not in ['MAX','MIN','LAST', 'AVG', 'SUM' ]:
                    print('''Aggregation not supported: Use one of these:
                            'MAX','MIN','LAST', 'AVG', 'SUM''')
                    return

                if window == 'm':
                    helper = f'''WITH helper AS(
                                    SELECT *, Substr(date({date_column}), 1,Instr(date({date_column}),'-')-1) AS year,
                                Substr(date({date_column}), -5,Instr(date({date_column}),'-')-3) AS month,
                                Substr(date({date_column}), -2,Instr(date({date_column}),'-')-1) AS day,
                                Substr(time({date_column}), 1,Instr(time({date_column}),':')-1) AS hour,
                                CAST(Substr(time({date_column}), -5,Instr(time({date_column}),':')-1)/{num} AS modu) AS mod
                                FROM {df_name}
                    )\n'''
                    groupby = 'year, month, day, hour, mod, '+str(id_column)

                if window == 'h':
                    helper = f'''WITH helper AS(
                                SELECT *, Substr(date({date_column}), 1,Instr(date({date_column}),'-')-1) AS year,
                                Substr(date({date_column}), -5,Instr(date({date_column}),'-')-3) AS month,
                                Substr(date({date_column}), -2,Instr(date({date_column}),'-')-1) AS day,
                                CAST(Substr(time({date_column}), 1,Instr(time({date_column}),':')-1)/{num} AS modu) as mod
                                FROM {df_name}
                    )\n'''
                    groupby = 'year, month, day, mod, '+str(id_column)

                if window == 'D':
                    helper = f'''WITH helper AS(
                      SELECT*, Substr(date({date_column}), 1,Instr(date({date_column}),'-')-1) AS year,
                                Substr(date({date_column}), -5,Instr(date({date_column}),'-')-3) AS month,
                                CAST(Substr(date({date_column}), -2,Instr(date({date_column}),'-')-1)/{num} AS modu) as mod
                                FROM {df_name}
                    )\n'''
                    groupby = 'year, month, mod, '+str(id_column)

        list_select = []
        for i in aggregation_per_col:
            aggElement = aggregation_per_col[i].upper()+'('+i+')' +' AS '+i
            list_select.append(aggElement)
        string_select = ',\n'.join(list_select)

        agg = 'SELECT '+ date_column+ ','+string_select + '\n FROM helper\n GROUP BY '+ groupby
        query = helper + agg

        return pysqldf(query)
    
    
       
    def add_seq(df, date_var, serie, freq, end_date='', start_date=''):
        """
        Creates a sequence of completes date/hours to a dataframe
        :params: dataframe in long format to add date/hour observations, date_var as string, 
            serie or id as string or list, freq as datetime.timedelta end and start date in format "%dd/%mm/%YYYY"
        :return: a Pandas dataframe
        """       
        
        df.loc[:, date_var] = df[date_var].apply(lambda x: x.tz_localize(None))

        if isinstance(serie, list)==False:
            seq = pd.DataFrame() 
            serie_list = list(df.loc[:, serie].unique())
            for i in serie_list:
                if start_date == '':
                    start_date = min(df.loc[df[serie]==i, date_var]).tz_localize(None)
                else:
                    start_date = pd.to_datetime(start_date, dayfirst=True).tz_localize(None)
                    
                if end_date == '':
                    end_date = max(df.loc[df[serie]==i, date_var]).tz_localize(None)
                else:
                    end_date = pd.to_datetime(end_date, dayfirst=True).tz_localize(None)
                                    
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
                        start_date = min(df.loc[(df[col_name]==id_col), date_var]).tz_localize(None)
                    else:
                        start_date = pd.to_datetime(start_date, dayfirst=True).tz_localize(None)
                        
                    if end_date == '':
                        end_date = max(df.loc[(df[col_name]==id_col), date_var]).tz_localize(None)
                    else:
                        end_date = pd.to_datetime(end_date, dayfirst=True).tz_localize(None)
                    
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
    
    def check_length_time_serie(df, date_var, index):
        """
        Checks the length that a time sequence of completes date/hours should have, so that it can be compared 
        with actual observation
        :params: df as pandas dataframe, date_var as string, index as list as groupby variable
        :return: a Pandas dataframe
        """       
        freq = pd.Series(df[date_var].unique()).dt.freq
        pivot = pd.pivot_table(df, index=index, values=date_var, aggfunc=['count', 'min', 'max']).reset_index()
        pivot.columns = pivot.columns.get_level_values(0)
        pivot.loc[:, 'td'] = pivot.loc[:, 'max'].max() - pivot.loc[:, 'min'].min()
        pivot.loc[:, 'count'] = pivot.loc[:, 'count'].astype(float)
        
        if freq=='H':
            pivot.loc[:, 'freq'] = 'H'
            pivot.loc[:, 'expected_obs'] = pivot.loc[:, 'td'].apply(lambda x: x.days*24) + pivot.loc[:, 'td'].apply(lambda x: x.seconds/3600) + 1
            pivot.loc[:, 'mismatch'] = 0
            pivot.loc[pivot['count']!=pivot['expected_obs'], 'mismatch'] = 1
            if sum(pivot.mismatch)>0:
                print('Expected length of sequence is NOT OK \n', pivot[[index, 'count', 'expected_obs']].drop_duplicates())
            else:
                print('Expected length of sequence is OK \n', pivot[[index, 'count', 'expected_obs']].drop_duplicates())

        elif freq=='D':
            pivot.loc[:, 'freq'] = 'D'
            pivot.loc[:, 'expected_obs'] = pivot.loc[:, 'td'].apply(lambda x: x.days) + pivot.loc[:, 'td'].apply(lambda x: x.seconds/3600*24) + 1
            pivot.loc[:, 'mismatch'] = 0
            pivot.loc[pivot['count']!=pivot['expected_obs'], 'mismatch'] = 1
            if sum(pivot.mismatch)>0:
                print('Expected length of sequence is NOT OK \n', pivot[[index, 'count', 'expected_obs']].drop_duplicates())
            else:
                print('Expected length of sequence is OK \n', pivot[[index, 'count', 'expected_obs']].drop_duplicates())
                
        else:
            pivot.loc[:, 'freq'] = np.nan
            pivot.loc[:, 'expected_obs'] = np.nan
            print('check_length_time_serie: could not infer frequency')


        return pivot
    
    def check_regressors_availability(df, date_var, regressors_list, forecast_end_date):
        """
        Checks the availability of regressors based on forecast end date
        :params: df as pandas dataframe, date_var as string, regressors_list as list and forecast_end_date as string in format "2022-12-31"
        :return: None
        """       
        forecast_end_date = pd.to_datetime(forecast_end_date, dayfirst = False)

        for r in regressors_list:
            if any(df.loc[df[date_var]<=forecast_end_date, r].isnull()):
                print('Latest filled available date for regressor', r, 'is', df.loc[df[r].isnull()==False, date_var].max(), '\n expected is', forecast_end_date)
                raise Exception('Regressor', r, 'shows null values <= forecast_end_date. \n Please, fill them before going on')
            else:
                print('Regressor', r, 'has all needed values')
        return None
    
    def remove_regressors_with_nan(df, date_var, regressors_list, forecast_end_date):
        """
        Remove regressors with nan based on forecast end date
        :params: df as pandas dataframe, date_var as string, regressors_list as list and forecast_end_date as string in format "2022-12-31"
        :return: pandas dataframe
        """       
        forecast_end_date = pd.to_datetime(forecast_end_date, dayfirst = False)
        
        for r in regressors_list:
            if any(df.loc[df[date_var]<=forecast_end_date, r].isnull()):
                print('Latest filled available date for regressor', r, 'is', df.loc[df[r].isnull()==False, date_var].max(), '\n expected is', forecast_end_date)
                print('Regressor', r, 'shows null values <= forecast_end_date. \n Regressor REMOVED')
                df.drop(columns = r, inplace=True)
            else:
                print('Regressor', r, 'has all needed values')
        return df
            
    def match_to_find(serie_to_find):
        """
        Finds a match in a list of possible words to match
        :params: serie_to_find as a list of words to match
        :return: a list
        """
        match_to_find = []
        match_to_find = match_to_find + [serie_to_find]
        match_to_find = match_to_find + [serie_to_find.lower()]
        match_to_find = match_to_find + [serie_to_find.upper()]
        match_to_find = match_to_find + [serie_to_find.capitalize()]
        match_to_find = match_to_find + [re.sub('[^a-zA-Z0-9 \n\.]', '_', serie_to_find)]
        match_to_find = match_to_find + [re.sub('[^a-zA-Z0-9 \n\.]', '_', serie_to_find.lower())]
        match_to_find = match_to_find + [re.sub('[^a-zA-Z0-9 \n\.]', '_', serie_to_find.upper())]
        match_to_find = match_to_find + [re.sub('[^a-zA-Z0-9 \n\.]', '_', serie_to_find.capitalize())]
        return match_to_find 
    
    def find_match(df, serie_name, match_to_find):
        """
        Finds a match in a dataframe serie given a list of possible words to match
        :params: dataframe, serie_name as string, match_to_find as a list of words to match
        :return: a list
        """

        list_to_match = list(df.loc[:, serie_name].unique())
        match_list = list()
        for m in match_to_find:
            match_list.extend([el for el in list_to_match if isinstance(el, collections_abc.Iterable) and (m in el)])

        match_list = list(dict.fromkeys(match_list))
        return match_list
    
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
    
    def id_outliers_IQR(df, q1, q3, date_var, id, var, freq_var):
        """
        Identifies outliers creatinga dummy variable (0/1) called outlier using IQR method, where quantile value can be set
        :param dates: dataframe, q1 and q3 values as numeric 0<x<1, date_var as string, var where we want to compute outliers as string,
        freq_var as string such as month or day
        :return: a Pandas dataframe
        """
        ### Removing negative values, since energy consumption can be only positive
        df = df.loc[df[var]>0, ].copy()
        
        if isinstance(id, 'list'):
            list_id = id + [var, freq_var]
        else:
            list_id = [id, var, freq_var]    
              
        # Freq var
        df.loc[:, freq_var] = df.loc[:, date_var].apply(lambda x: x.month)
        
        ### ID outliers
        grouped = df.loc[:, list_id].groupby(list_id)
        df_q1 = grouped.quantile(q1).reset_index()
        df_q1.rename(columns={var: 'q1'}, inplace=True)
        df_q3 = grouped.quantile(q3).reset_index()
        df_q3.rename(columns={var: 'q3'}, inplace=True)
        
        # Merge
        dfs = [df, df_q1, df_q3]
        df_outliers = reduce(lambda left,right: pd.merge(left,right,how='left', on=list_id, validate='m:1'), dfs)
       
        df_outliers.loc[:, 'IQR'] = df_outliers.q3 - df_outliers.q1
        df_outliers.loc[:, 'outlier'] = 0
        df_outliers.loc[((df_outliers[var]<(df_outliers.q1-1.5*df_outliers.IQR)) | (df_outliers[var]>(df_outliers.q3+1.5*df_outliers.IQR))), 'outlier']= 1
        var_cleaned = var + '_cleaned'
        df_outliers.loc[:, var_cleaned] = df_outliers.loc[:, var]
        df_outliers.loc[df_outliers.outlier==1, var_cleaned] = np.nan

        # Summarizing outliers in a pivot table
        pivot_sum = pd.pivot_table(df_outliers, values='outlier', index=list_id, aggfunc=sum).reset_index()
        pivot_len = pd.pivot_table(df_outliers, values='outlier', index=list_id, aggfunc=len).reset_index()
        pivot_len.rename(columns={'outlier': 'obs'}, inplace=True)
        pivot = pd.merge(pivot_sum, pivot_len, on=list_id, how='inner', validate='1:1')
        pivot.loc[:, 'outliers_perc'] =  round(pivot.outlier / pivot.obs,2)

        dict_outliers = {'df_outliers': df_outliers, 'pivot_outliers': pivot}
        return dict_outliers
        
   
class AlphabeticalCombinations:
    def write_neat_csv(saving_file, df_fcst):
        """
        Writes neat csv
        :params: saving_file as string, df_fcst as dataframe to write
        :return: None
        """
        df_fcst.to_csv(saving_file, sep=';', date_format="%Y-%m-%d %H:%M:%S", header=True, index=False, compression='infer', quoting=None, quotechar='"', doublequote=False, decimal='.')
               
        return(print('*** write_neat_csv: completed', saving_file))       
    
    def convert(string):
        """
        Convert string to list
        :params: string
        :return: a list
        """
        list1=[]
        list1[:0]=string
        return list1

    def excel_columns():
        """
        Counts excel columns
        :params: none
        :return: a list
        """
        alphabet_string = string.ascii_uppercase
        li = AlphabeticalCombinations.convert(alphabet_string)
        excel_columns = [letter for letter in alphabet_string]
        for L in li:
            aces = [L + li for li in li]
            excel_columns.extend(aces)

        return excel_columns

    def write_beautiful_excel(saving_file, dict_df_to_write):
        """
        Writes beautiful excel
        :params: saving_file as string, dict_df_to_write as dictionary with dict key as sheet name and dict value as data
        :return: None
        """
        ### Writing to Excel
        writer = pd.ExcelWriter(saving_file, engine='xlsxwriter', datetime_format='dd/mm/yyyy hh:mm:ss', date_format='dd/mm/yyyy')
        
        # FCST
        for d in list(dict_df_to_write.keys()):
            df = dict_df_to_write[d]
            df.to_excel(writer, sheet_name=d, index=False)

            # Make handles for workbook/sheet
            workbook = writer.book
            worksheet = writer.sheets[d]

            # Create positive/negative cell format
            format_simone = workbook.add_format({'num_format': '#,##0;- #,##0'})
            format_percentage = workbook.add_format({'num_format': '0.00%'})

            # Identify percentage columns
            cols_percentage = []
            for c in list(df.columns):
                try:
                    if any(df[c]>=1) and any(df[c]>=0) and any(df[c].between(0, 1, inclusive=False)):
                        cols_percentage.extend([c])
                except:
                    pass

            # Define the worksheet range to apply number format
            cols = AlphabeticalCombinations.excel_columns()
            row = len(df)
            format_range = '{}{}:{}{}'.format(cols[0], row, cols[len(df.columns)-1], row)

            # Apply number formats to specified range
            worksheet.set_column(format_range, None, format_simone)

            if len(cols_percentage)>0:
                for f in cols_percentage:
                    n = list(df.columns).index(f)
                    row = len(df)
                    format_range = '{}{}:{}{}'.format(cols[n], row, cols[n], row)
                    worksheet.set_column(format_range, None, format_percentage)

            #Iterate through each column and set the width == the max length in that column. A padding length of 2 is also added.
            for i, col in enumerate(df.columns):
                # find length of column i
                column_len = df[col].astype(str).str.len().max()
                # Setting the length if the column header is larger
                # than the max column value length
                column_len = max(column_len, len(col)) + 4
                # set the column length
                worksheet.set_column(i, i, column_len)

        ## Close the Pandas Excel writer and output the Excel file
        writer.save()
        return(print('*** write_beatiful_excel: completed', saving_file))

    def write_beautiful_excel_table(saving_file, dict_df_to_write):
        """
        Writes beautiful excel tables
        :params: saving_file as string, dict_df_to_write as dictionary with dict key as sheet name and dict value as data
        :return: None
        """
        ### Writing to Excel
        writer = pd.ExcelWriter(saving_file, engine='xlsxwriter', datetime_format='dd/mm/yyyy hh:mm:ss', date_format='dd/mm/yyyy')

        # FCST
        for d in list(dict_df_to_write.keys()):
            df = dict_df_to_write[d]
            df.to_excel(writer, sheet_name=d, index=False)

            # Make handles for workbook/sheet
            workbook = writer.book
            worksheet = writer.sheets[d]

            # Create positive/negative cell format
            format_simone = workbook.add_format({'num_format': '#,##0;- #,##0'})
            format_percentage = workbook.add_format({'num_format': '0.00%'})

            # Identify percentage columns
            cols_percentage = []
            for c in list(df.columns):
                try:
                    if any(df[c]>=1) and any(df[c]>=0) and any(df[c].between(0, 1, inclusive=False)):
                        cols_percentage.extend([c])
                except:
                    pass

            # Define the worksheet range to apply number format
            cols = AlphabeticalCombinations.excel_columns()
            row = len(df)
            format_range = '{}{}:{}{}'.format(cols[0], row, cols[len(df.columns)-1], row)

            # Apply number formats to specified range
            worksheet.set_column(format_range, None, format_simone)

            if len(cols_percentage)>0:
                for f in cols_percentage:
                    n = list(df.columns).index(f)
                    row = len(df)
                    format_range = '{}{}:{}{}'.format(cols[n], row, cols[n], row)
                    worksheet.set_column(format_range, None, format_percentage)

            #Iterate through each column and set the width == the max length in that column. A padding length of 2 is also added.
            for i, col in enumerate(df.columns):
                # find length of column i
                column_len = df[col].astype(str).str.len().max()
                # Setting the length if the column header is larger
                # than the max column value length
                column_len = max(column_len, len(col)) + 4
                # set the column length
                worksheet.set_column(i, i, column_len)

            # Create a list of column headers, to use in add_table().
            column_settings = []
            for header in df.columns:
                column_settings.append({'header': header})

            # Add the table.
            worksheet.add_table(0, 0, df.shape[0], df.shape[1] - 1, {'columns': column_settings})

        ## Close the Pandas Excel writer and output the Excel file
        writer.save()
        return(print('*** write_beatiful_excel: completed', saving_file))
    
