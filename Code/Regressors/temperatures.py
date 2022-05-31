# selenium for web driving
from logging import raiseExceptions
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

# time for pausing between navigation
import time
import glob
import shutil

# datetime functions
import datetime as dt

# file management functions
import os
import configparser
import ctypes

# data elaboration functions
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from functools import reduce

# custom functions
from Code.Utils.utils import Utils, AlphabeticalCombinations

class Temperatures:
        
    def ten_year(df, id, date_var, freq, temperature_list, start_date, end_date):
        """
        Computes ten year temperatures and asis temperatures
        :params: dataframe
        :return: a Pandas dataframe, a .pkl file and a .xlsx file
        """
        ten_year_list = []
        ten_year_overall_list = []
        for t in temperature_list:
            ten_year_list = ten_year_list + [t + '_ten_year']
            ten_year_overall_list = ten_year_overall_list + [t + '_ten_year_overall']
            
        df_seq = Utils.add_seq(df, date_var = date_var, serie=id, freq = freq, start_date=start_date, end_date=end_date)
        df_seq.loc[:, 'months_days'] = df_seq.loc[:, date_var].dt.strftime('%m/%d')
        
        # Defining averages by id
        df_to_merge = pd.pivot_table(df_seq, values=temperature_list, index=[id, 'months_days'], aggfunc=np.mean).reset_index()
        col_list = [id, 'months_days'] + ten_year_list
        df_to_merge.columns = col_list
        
        # Defining overall averages
        df_to_merge_overall = pd.pivot_table(df_seq, values=temperature_list, index=['months_days'], aggfunc=np.mean).reset_index()
        col_list_overall = ['months_days'] + ten_year_overall_list 
        df_to_merge_overall.columns = col_list_overall
        
        # Merging
        df_merge = pd.merge(df_seq, df_to_merge, on=[id, 'months_days'], how='left', validate='m:1')
        df_merge_overall = pd.merge(df_merge, df_to_merge_overall, on=['months_days'], how='left', validate='m:1')
               
        ### Creating As-Is temperatures: where available use actual temp, if not use ten year
        for t in temperature_list:
            asis_name = t + '_asis'
            ten_year_name = t + '_ten_year'
            ten_year_overall_name = t + '_ten_year_overall'
            df_merge_overall.loc[:, asis_name] = df_merge_overall.loc[:, t]
            df_merge_overall.loc[df_merge_overall[asis_name].isnull(), asis_name] = df_merge_overall.loc[:, ten_year_name]
            df_merge_overall.loc[df_merge_overall[asis_name].isnull(), asis_name] = df_merge_overall.loc[:, ten_year_overall_name]

            if (any(df_merge_overall[asis_name].isnull())):
                print('ten_year: asis temperatures still CONTAIN nan value: removing')
                df_merge_overall = df_merge_overall.loc[df_merge_overall[asis_name].isnull()==False, ]
            else:
                print('ten_year: asis temperatures do NOT contain any nan value')
                
        df_ten_year = df_merge_overall.loc[:, ['site_id', 'timestamp', 'temperature', 'distance', 'months_days',
       'temperature_ten_year', 'temperature_asis']]

        return df_ten_year
    
   
