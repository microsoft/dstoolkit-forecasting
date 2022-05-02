# data elaboration functions
import pandas as pd
import numpy as np
import re

# file management functions
import os
import glob
import holidays as h

# time management functions
import datetime as dt

# custom functions
from Configuration.config import cfg_path
from Code.Utils.utils import Utils

class Regressors:
    def create_interactions(df, var1, var2):
        """
        Adds interaction terms between two variables as var1*var2 to dataframe
        :params: dataframe, var1 and var 2 as string
        :return: a Pandas dataframe
        """
        variables = df[[var1, var2]]
        for i in range(0, variables.columns.size):
            for j in range(0, variables.columns.size):
                col1 = variables.columns[i]
                col2 = variables.columns[j]
                if i <= j:
                    name = col1 + "*" + col2
                    df.loc[:, name] = variables[col1] * variables[col2]

        df.drop(columns = [var1 + "*" + var1], inplace=True)
        df.drop(columns = [var2 + "*" + var2], inplace=True)
        return df
        
    def create_non_linear_terms(df, var, n):
        """
        Adds non linear terms as var^2 to dataframe
        :params: dataframe, var as string and n as int
        :return: a Pandas dataframe
        """
        name = var + "^" + str(n)
        df.loc[:, name] = df.loc[:, var]**n
        return df
    
    def add_holidays_by_country(df, date_var, country):
        """
        Adds holidays a dummy variable (0/1) to dataframe
        :params: dataframe, date_var as string, country as string
        :return: a Pandas dataframe
        """
        if 'holidays' in list(df.columns):
            print('add_holidays_by_country: holidays column already present')
        else:
            holidays = eval("h." + country.capitalize() + "()")
            date_holidays = df.loc[:, date_var].apply(lambda x: int(1) if x in holidays else int(0))
            date_holidays = pd.DataFrame(date_holidays)
            date_holidays.columns = pd.Index(['holidays'])
            df = pd.concat([df, date_holidays], axis=1)
        return df
        
    def add_weekdays(df, date_var):
        """
        Adds weekdays a dummy variables (0/1) for each weekday to dataframe
        :params: dataframe, date_var as string
        :return: a Pandas dataframe
        """
        df.loc[:,'wd_mon'] = df.loc[:, date_var].apply(lambda x: int(1) if x.weekday() == 0 else int(0))
        df.loc[:,'wd_tue'] = df.loc[:, date_var].apply(lambda x: int(1) if x.weekday() == 1 else int(0))
        df.loc[:,'wd_wed'] = df.loc[:, date_var].apply(lambda x: int(1) if x.weekday() == 2 else int(0))
        df.loc[:,'wd_thu'] = df.loc[:, date_var].apply(lambda x: int(1) if x.weekday() == 3 else int(0))
        df.loc[:,'wd_fri'] = df.loc[:, date_var].apply(lambda x: int(1) if x.weekday() == 4 else int(0))
        df.loc[:,'wd_sat'] = df.loc[:, date_var].apply(lambda x: int(1) if x.weekday() == 5 else int(0))
        df.loc[:,'wd_sun'] = df.loc[:, date_var].apply(lambda x: int(1) if x.weekday() == 6 else int(0))
        return df

    def add_months(df, date_var):
        """
        Adds months a dummy variables (0/1) for each month to dataframe
        :params: dataframe, date_var as string
        :return: a Pandas dataframe
        """
        for i in range(1, 13):
            if i < 10:
                varname = 'month_0' + str(i)
            else:
                varname = 'month_' + str(i)
            
            df.loc[:, varname] = df.loc[:, date_var].apply(lambda x: int(1) if x.month == i else int(0))
        return df 
        
    def calculate_degree_days(df, base_temperature, temperature):
        """Calculate the Degree Days Heating and Cooling values
        :params: dataframe, base temperature to start and actual temperature as string
        :return: a pandas dataframe
        """
        df['DDC_temperature'] = (df[temperature] - df[base_temperature]).clip(lower=0)
        df['DDH_temperature'] = (df[base_temperature] - df[temperature]).clip(lower=0)

        return df

    def merge_holidays_by_date(df, df_holidays, id):
        """Merge Holiday df with the train df
        :params: df as dataframe, df_holidays as df containing info on holidays, id as string
        :return: a pandas dataframe
        """
        date_var = Utils.find_date(df)
        date_var_holidays = Utils.find_date(df_holidays)
        
        cols_to_keep = list(df.columns)
        
        df['date_key'] = df[date_var].dt.year.astype(str) + df[date_var].dt.month.astype(str) + df[date_var].dt.day.astype(str)
        df_holidays['date_key'] = df_holidays[date_var_holidays].dt.year.astype(str) + df_holidays[date_var_holidays].dt.month.astype(str) + df_holidays[date_var_holidays].dt.day.astype(str)
        
        df.loc[:, 'holidays'] = int(0)
        df_merge = pd.merge(df, df_holidays, how="left", on=["date_key", id], indicator=True)
        df_merge.loc[df_merge._merge=='both', 'holidays'] = int(1)
        
        cols_to_keep = cols_to_keep + ['holidays']
        df = df_merge[cols_to_keep].copy()

        return df

    def merge_additional_days_off(df, df_metadata, id, dict_days_off):
        """Merge Site Weekend data with train df
        :params: df as dataframe, df_metadata as df containing additional info, id as string, dict_days_off as dictionary 
        :return: a pandas dataframe
        """
        date_var = Utils.find_date(df)

        # Sites only had weekly leaves on Friday, Saturday and Sunday
        list_days_off = list(dict_days_off.keys())
        df.loc[:, 'day_off'] = int(0)
        for d in list_days_off: 
            leave = (df[date_var].dt.dayofweek == dict_days_off[d]) & (df[id].isin(df_metadata[df_metadata[d]][id]))
            df.loc[leave==True, 'day_off'] = int(1)

        df['day_off'] = df['day_off'].astype("int8")

        return df

    def merge_weather(df, weather, date_var, id):
        """Merge weather data into the train df
        :params: df as dataframe, weather as dataframe with weather info, date_var as string, id as string
        :return: a pandas dataframe
        
        """
        
        date_var = Utils.find_date(df)
        date_var_weather = Utils.find_date(weather)

        # drop duplicate values in weather and pick the closest weather station
        weather_cleaned = weather.sort_values([date_var, id, "distance"]).groupby([date_var, id]).first().reset_index()
        assert weather_cleaned.groupby([date_var, id]).count().max().max() == 1

        df = pd.merge(df.sort_values([date_var, id]), weather_cleaned.sort_values([date_var_weather]), left_on=[date_var, id], right_on= [date_var_weather, id], how='left', validate="m:1")

        return df