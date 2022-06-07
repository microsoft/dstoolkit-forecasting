import pandas as pd

from Code.Utils.utils import Utils
from Code.Scoring.train_test import TrainTest

class TimeSeriesDataset():
    def __init__(self, df, y_col, datetime_col = None, sku_col = None, regressor_cols = []) -> None:
        self.df = df.copy()
        self.y_col = y_col

        self.group_by_cols = [sku_col] if sku_col != None else []
        self.datetime_col = Utils.find_date(df) if datetime_col == None else datetime_col

        self.regressors = regressor_cols if len(regressor_cols) else set(df.columns) - set(self.group_by_cols) - set(self.datetime_col) - {y_col}

    
    def add_weekdays_dummies(self) -> list:
        """
        Adds weekdays a dummy variables (0/1) for each weekday to dataframe
        :params: dataframe, self.datetime_col as string
        :return: a Pandas dataframe
        """
        self.df.loc[:,'wd_mon'] = self.df.loc[:, self.datetime_col].apply(lambda x: int(1) if x.weekday() == 0 else int(0))
        self.df.loc[:,'wd_tue'] = self.df.loc[:, self.datetime_col].apply(lambda x: int(1) if x.weekday() == 1 else int(0))
        self.df.loc[:,'wd_wed'] = self.df.loc[:, self.datetime_col].apply(lambda x: int(1) if x.weekday() == 2 else int(0))
        self.df.loc[:,'wd_thu'] = self.df.loc[:, self.datetime_col].apply(lambda x: int(1) if x.weekday() == 3 else int(0))
        self.df.loc[:,'wd_fri'] = self.df.loc[:, self.datetime_col].apply(lambda x: int(1) if x.weekday() == 4 else int(0))
        self.df.loc[:,'wd_sat'] = self.df.loc[:, self.datetime_col].apply(lambda x: int(1) if x.weekday() == 5 else int(0))
        self.df.loc[:,'wd_sun'] = self.df.loc[:, self.datetime_col].apply(lambda x: int(1) if x.weekday() == 6 else int(0))
        
        weekdays_dummies_cols = ['wd_fri', 'wd_mon', 'wd_tue', 'wd_sat', 'wd_sun', 'wd_thu']
        return weekdays_dummies_cols
    
    def add_month_dummies(self) -> list:
        """
        Adds months a dummy variables (0/1) for each month to dataframe
        :params: dataframe, date_var as string
        :return: a Pandas dataframe
        """
        month_dummies_cols = []
        for i in range(1, 13):
            if i < 10:
                varname = 'month_0' + str(i)
            else:
                varname = 'month_' + str(i)
            
            month_dummies_cols.append(varname)
            self.df.loc[:, varname] = self.df.loc[:, self.datetime_col].apply(lambda x: int(1) if x.month == i else int(0))
        return month_dummies_cols

    def add_interactions(self, col1 = [], col2 = []) -> list:
        """
        Adds interaction terms between two variables as var1*var2 to dataframe
        :params: dataframe, var1 and var 2 as string
        :return: a Pandas dataframe
        """
        variables = self.df[[col1, col2]]
        interaction_cols = []
        for i in range(0, variables.columns.size):
            for j in range(0, variables.columns.size):
                col1 = variables.columns[i]
                col2 = variables.columns[j]
                if i <= j:
                    name = col1 + "*" + col2
                    interaction_cols.append(name)
                    self.df.loc[:, name] = variables[col1] * variables[col2]

        self.df.drop(columns = [col1 + "*" + col1], inplace=True)
        self.df.drop(columns = [col2 + "*" + col2], inplace=True)
        
        return interaction_cols

    def add_non_linear_term(self, col, power) -> list:
        """
        Adds non linear terms as var^2 to dataframe
        :params: dataframe, var as string and n as int
        :return: a Pandas dataframe
        """
        name = col + "^" + str(power)
        self.df.loc[:, name] = self.df.loc[:, col]**power
        return [name]

    def add_holidays(self, holidays_df = None, country = None):
        """
        Adds holidays a dummy variable (0/1) to dataframe
        :params: dataframe, date_var as string, country as string
        :return: a Pandas dataframe
        """
        if 'holidays' in list(self.df.columns):
            print('add_holidays_by_country: holidays column already present')
        else:
            date_holidays = self.df.loc[:, self.datetime_col].apply(lambda x: int(1) if x in holidays_df else int(0))
            date_holidays = pd.DataFrame(date_holidays)
            date_holidays.columns = pd.Index(['holidays'])
            self.df = pd.concat([self.df, date_holidays], axis=1)
        return self.df

    def merge_weather(self, weather, id):
        """Merge weather data into the train df
        :params: df as dataframe, weather as dataframe with weather info, date_var as string, id as string
        :return: a pandas dataframe
        
        """
        date_var_weather = Utils.find_date(weather)

        # drop duplicate values in weather and pick the closest weather station
        weather_cleaned = weather.sort_values([self.datetime_col, id, "distance"]).groupby([self.datetime_col, id]).first().reset_index()
        assert weather_cleaned.groupby([self.datetime_col, id]).count().max().max() == 1

        df = pd.merge(df.sort_values([self.datetime_col, id]), weather_cleaned.sort_values([date_var_weather]), left_on=[self.datetime_col, id], right_on= [date_var_weather, id], how='left', validate="m:1")

        return df

    def stats_per_sku(self):
        """
        Helper function to identify amount of data per site
        :params: df as pandas dataframe, id as string, date_var as string
        :return: a pandas dataframe
        """
        return pd.DataFrame(
            [{
                "id": site, 
                "Years": self.df.loc[(self.df[id] == site), self.datetime_col].dt.year.unique(), 
                "Max Timestamp": self.df.loc[(self.df[id] == site), self.datetime_col].max(), 
                "Min Timestamp": self.df.loc[(self.df[id] == site), self.datetime_col].min(),
                "Samples": self.df[(self.df[id] == site)].count().sum()
                } for site in self.df[self.group_by_cols].unique()]
        ).sort_values("Samples", ascending=False)

    def get_train_test_split(self, test_start_date, test_end_date = None, train_start_date = None, train_end_date = None, forecast_scope=720):
        if test_end_date=='':            
            test_end_date =  self.df.loc[:, self.datetime_col].max().strftime('%Y-%m-%d')
        else:
            test_end_date = pd.to_datetime(test_end_date, format='%Y-%m-%d')
            
        if test_start_date=='':
            test_start_date = (pd.to_datetime(test_end_date, format='%Y-%m-%d') - pd.DateOffset(days=365)).strftime('%Y-%m-%d')
        else:
            test_start_date = pd.to_datetime(test_start_date, format='%Y-%m-%d')
            
        # Train set: set train set from test start date -1 to minimum date available
        if train_start_date=='':   
            train_start_date = self.df.loc[:, self.datetime_col].min().strftime('%Y-%m-%d')
        else:
            train_start_date = pd.to_datetime(train_start_date, format='%Y-%m-%d')
            
        if train_end_date=='':
            train_end_date = (pd.to_datetime(test_start_date, format='%Y-%m-%d') - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        else:
            train_end_date = pd.to_datetime(train_end_date, format='%Y-%m-%d')

        dict_train = TrainTest.def_train(self.df, self.y_col, [16, 22, 25, 42, 49], train_start_date, train_end_date)
        dict_test = TrainTest.def_test(self.df, self.y_col, [16, 22, 25, 42, 49], test_start_date, test_end_date, forecast_scope)

        train_ds = TimeSeriesDataset(dict_train, self.y_col, self.group_by_cols, self.datetime_col, self.regressors)
        test_ds = TimeSeriesDataset(dict_test, self.y_col, self.group_by_cols, self.datetime_col, self.regressors)
        return train_ds, test_ds

    def plot(self, y, chart_title):
        chart_title = 'Energy prediction'
        y = 'value'
        for s in list(df_pbi[id].unique()):
            df_plot = df_pbi.loc[df_pbi[id]==s, ]
            saving_name = str(id) + '_' + str(s) + '_energy_prediction'
            plot = Plots.sliding_fcst_plot(df_plot, y, 'fcst', chart_title, kpi=True)
            df_plot.to_csv(os.path.join(root, cfg_path.data_dir.output_path, saving_name + ".csv"))
            plot.write_html(os.path.join(root, cfg_path.data_dir.plot_path, saving_name + ".html"))

    def enh_idclass5(self, threshold, perc, quant, highest, lowest):
        pass
    
    def classify_intermittent(self, type='mix_floor_Q_999', thres_cv2_constant=0.01, thres_cv2=2, thres_adi=3, thres_sddi=6.2, min_time_cons=2):
        ''' Classifies intermittent time series based on indicator values
        :params: df as pandas dataframe, type as string, thres_cv2_constant as numeric, thres_cv2 as numeric, thres_adi as numeric, thres_sddi as numeric, min_time_cons as numeric
        :return: a pandas dataframe
        '''
        # Excluding the ids for which the indicators are np.nan
        score_no_nan = self.df.dropna()

        # Regular
        mask_regular = (score_no_nan.type == type) &\
                                        (score_no_nan.k > min_time_cons) &\
                                        (score_no_nan.cv2 >= thres_cv2_constant) &\
                                        (score_no_nan.cv2 < thres_cv2) &\
                                        (score_no_nan.cv2 < thres_adi) &\
                                        (score_no_nan.cv2 < thres_sddi)
        df_regular = score_no_nan.loc[mask_regular, ]
        try:
            df_regular.loc[:, 'profile'] = 'regular'
            print('classify_intermittent: regular ids', len(df_regular))
        except:
            print('classify_intermittent: no regular ids')

        # Constant at zero
        mask_constant_zero = (score_no_nan.type == type) &\
                                        (score_no_nan.k <= min_time_cons)
        df_constant_zero = score_no_nan.loc[mask_constant_zero, ]
        try:
            df_constant_zero.loc[:, 'profile'] = 'constant_zero'
            print('classify_intermittent: constant_zero ids', len(df_constant_zero))
        except:
            print('classify_intermittent: no constant_zero ids')

        # Constant
        mask_constant = (score_no_nan.type == type) &\
                                        (score_no_nan.k > min_time_cons) &\
                                        (score_no_nan.cv2 < thres_cv2_constant) &\
                                        (score_no_nan.cv2 < thres_adi) &\
                                        (score_no_nan.cv2 < thres_sddi)
        df_constant = score_no_nan.loc[mask_constant, ]
        try:
            df_constant.loc[:, 'profile'] = 'constant'
            print('classify_intermittent: constant ids', len(df_constant))
        except:
            print('classify_intermittent: no constant ids')

        # Intermittent
        mask_intermittent = (score_no_nan.type == type) &\
                                        (score_no_nan.k > min_time_cons) &\
                                        (score_no_nan.cv2 < thres_cv2) &\
                                        (score_no_nan.cv2 >= thres_adi) &\
                                        (score_no_nan.cv2 < thres_sddi)
        df_intermittent = score_no_nan.loc[mask_intermittent, ]
        try:
            df_intermittent.loc[:, 'profile'] = 'intermittent'
            print('classify_intermittent: intermittent ids', len(df_intermittent))
        except:
            print('classify_intermittent: no intermittent ids')

        # Lumpy
        mask_lumpy = (score_no_nan.type == type) &\
                                        (score_no_nan.k > min_time_cons) &\
                                        (score_no_nan.cv2 >= thres_cv2) &\
                                        (score_no_nan.cv2 >= thres_adi) &\
                                        (score_no_nan.cv2 < thres_sddi)
        df_lumpy = score_no_nan.loc[mask_lumpy, ]
        try:    
            df_lumpy.loc[:, 'profile'] = 'lumpy'
            print('classify_intermittent: lumpy', len(df_lumpy))
        except:
            print('classify_intermittent: no lumpy ids')

        # Erratic
        mask_erratic = (score_no_nan.type == type) &\
                                        (score_no_nan.k > min_time_cons) &\
                                        (score_no_nan.cv2 >= thres_cv2) &\
                                        (score_no_nan.cv2 < thres_adi) &\
                                        (score_no_nan.cv2 < thres_sddi)
        df_erratic = score_no_nan.loc[mask_erratic, ]
        try:
            df_erratic.loc[:, 'profile'] = 'erratic'
            print('classify_intermittent: erratic ids', len(df_erratic))
        except:
            print('classify_intermittent: no erratic ids')

        # Unforecastable time
        mask_unforecastable_time = (score_no_nan.type == type) &\
                                        (score_no_nan.k > min_time_cons) &\
                                        (score_no_nan.cv2 < thres_cv2) &\
                                        (score_no_nan.cv2 >= thres_sddi)
        df_unforecastable_time = score_no_nan.loc[mask_unforecastable_time, ]
        try:
            df_unforecastable_time.loc[:, 'profile'] = 'unforecastable_time'
            print('classify_intermittent: unforecastable_time ids', len(df_unforecastable_time))
        except:
            print('classify_intermittent: no unforecastable_time ids')

        # Unforecastable quantity
        mask_unforecastable_quantity = (score_no_nan.type == type) &\
                                        (score_no_nan.k > min_time_cons) &\
                                        (score_no_nan.cv2 >= thres_cv2) &\
                                        (score_no_nan.cv2 >= thres_sddi)
        df_unforecastable_quantity = score_no_nan.loc[mask_unforecastable_quantity, ]
        try:
            df_unforecastable_quantity.loc[:, 'profile'] = 'unforecastable_quantity'
            print('classify_intermittent: unforecastable_quantity ids', len(df_unforecastable_quantity))
        except:
            print('classify_intermittent: no unforecastable_quantity ids')
        
        # df_profiling        
        df_profiling = pd.concat([df_regular, df_constant_zero, df_constant, df_intermittent, df_lumpy, df_erratic, df_unforecastable_time, df_unforecastable_quantity], axis=0)
        
        return df_profiling


