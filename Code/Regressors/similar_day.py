# data elaboration functions
import numpy as np
import pandas as pd
import holidays as h
from functools import reduce

# datetime functions
import dateutil
import datetime
from dateutil.relativedelta import relativedelta

# custom functions
from Code.Regressors.regressors import Regressors
from Code.Utils.utils import AlphabeticalCombinations, Utils

class SimilarDay: 
    def get_similar_days_in_previous_year(dates, country):
        """
        Retrieves the similar day for a given date. 
        :param dates: a list-like object of dates, country as string
        :return: a Pandas series of similar days
        """
        d = pd.to_datetime(pd.Series(dates))        
        holidays = eval("h." + country.capitalize() + "()")
        return d.apply(lambda x: SimilarDay.get_similar_day_in_previous_year(x, holidays))

    def get_similar_days_in_previous_week(dates, country):
        """
        Retrieves the similar day for a given date.
        :param dates: a list-like object of dates, country as string
        :return: a Pandas series of similar days
        """
        d = pd.to_datetime(pd.Series(dates))
        holidays = eval("h." + country.capitalize() + "()")
        return d.apply(lambda x: SimilarDay.get_similar_day_in_previous_week(x, holidays))


    def get_similar_day_in_previous_year(d, holiday_calendar):
        """
        Retrieves the similar day for a given date. If the given date is not an holiday, the similar day is the
        closest day of the previous year in terms of calendar position which shares the weekday. If such a date is an holiday,
        the same weekday of the week before is considered. 
        If the given date is an holiday, its similar day is the closest holiday to the given date in the previous year.
        :param d: a date
        :param holiday_calendar: a calendar from holidays package
        :return: the similar day
        """
        if not d or pd.isna(d):
            return None

        new_date = d - relativedelta(years=1)
        holiday = holiday_calendar.get(d)
        diff = d.weekday() - new_date.weekday() if d.weekday() >= new_date.weekday() \
            else d.weekday() - new_date.weekday() + 7

        if not holiday:
            new_date = new_date + datetime.timedelta(days=diff)
            while holiday_calendar.get(new_date):
                new_date = new_date - datetime.timedelta(days=7)
        # elif holiday == 'Pasqua di Resurrezione':
        #     new_date = dateutil.easter.easter(new_date.year)
        # elif holiday == "Lunedì dell'Angelo":
        #     new_date = dateutil.easter.easter(new_date.year) + datetime.timedelta(days=1)

        return new_date

    def get_similar_day_in_previous_week(d, holiday_calendar):
        """
        Retrieves the similar day for a given date. If the given date is not an holiday, the similar day is the
        closest day of the previous year in terms of calendar position which shares the weekday. If such a date is an holiday,
        the same weekday of the week before is considered. 
        If the given date is an holiday, its similar day is the closest holiday to the given date in the previous year.
        :param d: a date
        :param holiday_calendar: a calendar from holidays package
        :return: the similar day
        """
        if not d or pd.isna(d):
            return None

        new_date = d - relativedelta(weeks=1)
        holiday = holiday_calendar.get(d)
        diff = d.weekday() - new_date.weekday() if d.weekday() >= new_date.weekday() \
            else d.weekday() - new_date.weekday() + 7

        if not holiday:
            new_date = new_date + datetime.timedelta(days=diff)
            while holiday_calendar.get(new_date):
                new_date = new_date - datetime.timedelta(days=7)
        # elif holiday == 'Pasqua di Resurrezione':
        #     new_date = dateutil.easter.easter(new_date.year)
        # elif holiday == "Lunedì dell'Angelo":
        #     new_date = dateutil.easter.easter(new_date.year) + datetime.timedelta(days=1)

        return new_date

class StandardConsumption:   
    def get_standard_consumption_as_mean(df, id, date_var, var, country):
        """
        Retrieves the standard consumption for a given date as hourly monthly mean differentiated by holiday, weekend, weekdays. 
        :params: dataframe and date_var as string, var as string, country as string
        :return: the similar day
        """

        df = Regressors.add_holidays_by_country(df, date_var, country)
        df = Regressors.add_weekdays(df, date_var)
        df.loc[:, 'day'] = df.loc[:, date_var].dt.day
        df.loc[:, 'hour'] = df.loc[:, date_var].dt.hour
        df.loc[:, 'month'] = df.loc[:, date_var].dt.month
        
        timedelta = Utils.delta_format(abs(np.diff(df[date_var])).mean())
        freq = Utils.find_freq(timedelta)
        
        if freq == 'D':
            freq_var='day'
        else:
            freq_var='hour'
        
        # Compute standard consumption as means        
        mask = (~df[var].isnull()) &  ((df.wd_mon==1) | (df.wd_tue==1) | (df.wd_wed==1) | (df.wd_thu==1) | (df.wd_fri==1)) & (df.holidays==0) 
        df_mean_weekdays = pd.pivot_table(df.loc[mask==True, ], index=[id, 'month', freq_var], values=var, aggfunc=np.mean).reset_index()
        new_var = var + '_std_weekdays'
        df_mean_weekdays.rename(columns={var: new_var}, inplace=True)
        df_mean_weekdays.loc[df_mean_weekdays[new_var]<0, new_var] = 0
        
        mask = (~df[var].isnull()) & ((df.wd_sat==1) | (df.wd_sun==1)) & (df.holidays==0) 
        df_mean_weekend = pd.pivot_table(df.loc[mask==True, ], index=[id, 'month', freq_var], values=var, aggfunc=np.mean).reset_index()
        new_var = var + '_std_weekend'
        df_mean_weekend.rename(columns={var: new_var}, inplace=True)
        df_mean_weekend.loc[df_mean_weekend[new_var]<0, new_var] = 0
        
        mask = (~df[var].isnull()) & (df.holidays==1) 
        df_mean_holidays = pd.pivot_table(df.loc[mask==True, ], index=[id, 'month', freq_var], values=var, aggfunc=np.mean).reset_index()
        new_var = var + '_std_holidays'
        df_mean_holidays.rename(columns={var: new_var}, inplace=True)
        df_mean_holidays.loc[df_mean_holidays[new_var]<0, new_var] = 0
        
        # Merging
        dfs = [df_mean_holidays, df_mean_weekdays, df_mean_weekend]
        df_mean = reduce(lambda left,right: pd.merge(left,right,how='outer', on=[id, 'month', freq_var], validate='1:1'), dfs)
        df = pd.merge(df, df_mean, how='left', on=[id, 'month', freq_var], validate='m:1')
        
        return df
    
    
    def get_minimum_consumption(df, date_var, var, country):
        """
        Retrieves the minimum consumption for a given date as hourly monthly minimum value differentiated by holiday, weekend, night. 
        :params: dataframe and date_var as string, var as string, country as string
        :return: the similar day
        """

        df = Regressors.add_holidays_by_country(df, date_var, country)
        df = Regressors.add_weekdays(df, date_var)
        df.loc[:, 'day'] = df.loc[:, date_var].dt.day
        df.loc[:, 'hour'] = df.loc[:, date_var].dt.hour
        df.loc[:, 'month'] = df.loc[:, date_var].dt.month
        
        timedelta = Utils.delta_format(abs(np.diff(df[date_var])).mean())
        freq = Utils.find_freq(timedelta)
        
        if freq == 'D':
            freq_var='day'
        else:
            freq_var='hour'
        
        # Compute min consumption        
        mask = (~df[var].isnull()) & (df.holidays==0) & ((df.wd_mon==1) | (df.wd_tue==1) | (df.wd_wed==1) | (df.wd_thu==1) | (df.wd_fri==1))
        df_min_weekdays = pd.pivot_table(df.loc[mask==True, ], index=[id, 'month', freq_var], values=var, aggfunc=np.min).reset_index()
        new_var = var + '_min_weekdays'
        df_min_weekdays.rename(columns={var: new_var}, inplace=True)
        df_min_weekdays.loc[df_min_weekdays[new_var]<0, new_var] = 0
        
        mask = (~df[var].isnull()) & ((df.wd_sat==1) | (df.wd_sun==1)) & (df.holidays==0) 
        df_min_weekend = pd.pivot_table(df.loc[mask==True, ], index=[id, 'month', freq_var], values=var, aggfunc=np.min).reset_index()
        new_var = var + '_min_weekend'
        df_min_weekend.rename(columns={var: new_var}, inplace=True)
        df_min_weekend.loc[df_min_weekend[new_var]<0, new_var] = 0
        
        mask = (~df[var].isnull()) & (df.holidays==1) 
        df_min_holidays = pd.pivot_table(df.loc[mask==True, ], index=[id, 'month', freq_var], values=var, aggfunc=np.min).reset_index()
        new_var = var + '_min_holidays'
        df_min_holidays.rename(columns={var: new_var}, inplace=True)
        df_min_holidays.loc[df_min_holidays[new_var]<0, new_var] = 0
        
        # Merging
        dfs = [df_min_holidays, df_min_weekdays, df_min_weekend]
        df_min = reduce(lambda left,right: pd.merge(left,right,how='outer', on=[id, 'month', freq_var], validate='1:1'), dfs)
        df = pd.merge(df, df_min, how='left', on=[id, 'month', freq_var], validate='m:1')
        
        return df


