# data elaboration functions
import pandas as pd
import numpy as np

# statistical functions
from scipy.stats.mstats import winsorize

class Intermittent:
    def cv2_by_group(df, y, grouping_var, highest=0.05, lowest=0.05):
        ''' Computes cv2 by group
        :params: df as pandas dataframe, y as string, grouping_var as list, highest and lowest as scalars 0<=x<=1 as winsorization percentages
        :return: a dataframe'''
        cv2_by_freq = df.loc[:, [grouping_var, y]].groupby(grouping_var).apply(lambda x: Intermittent.cv2(x, highest, lowest)).reset_index(level=grouping_var)
        cv2_by_freq.columns = [grouping_var, 'cv2_by_group']
        return cv2_by_freq
    
    def cv2(array, highest=0.05, lowest=0.05):
        ''' Winsorization is the process of replacing the extreme values of statistical data in order to limit 
        the effect of the outliers on the calculations or the results obtained by using that data. 
        The mean value calculated after such replacement of the extreme values is called winsorized mean.
        :params: array as numpy array, highest and lowest as scalars 0<=x<=1 as winsorization percentages
        :return: a scalar'''
        winsorized_array = winsorize(array,(highest,lowest))
        cv2 = (np.std(winsorized_array)/np.mean(winsorized_array))**2        
        return cv2
    
    def adi(array, highest=0.05, lowest=0.05):
        ''' Winsorization is the process of replacing the extreme values of statistical data in order to limit 
        the effect of the outliers on the calculations or the results obtained by using that data. 
        The mean value calculated after such replacement of the extreme values is called winsorized mean.
        :params: array as numpy array, highest and lowest as scalars 0<=x<=1 as winsorization percentages
        :return: a scalar'''
        winsorized_array = winsorize(array,(highest,lowest))
        adi = np.mean(winsorized_array)
        return adi
    
    def sddi(array, highest=0.05, lowest=0.05):
        ''' Winsorization is the process of replacing the extreme values of statistical data in order to limit 
        the effect of the outliers on the calculations or the results obtained by using that data. 
        The mean value calculated after such replacement of the extreme values is called winsorized mean.
        :params: array as numpy array, highest and lowest as scalars 0<=x<=1 as winsorization percentages
        :return: a scalar'''
        winsorized_array = winsorize(array,(highest,lowest))
        sddi = np.std(winsorized_array)
        return sddi
    
    def idclass3(vect, threshold, perc, quant, highest, lowest):
        ''' Computes indicator values
        :params: vect as numpy array, threshold as numeric, perc as numeric, quant as numeric, highest and lowest as scalars 0<=x<=1 as winsorization percentages
        :return: a dictionary
        '''  
        
        if isinstance(vect,(np.ndarray))==False:
            try:
                vect = np.array(vect)
            except:        
                raise Exception("identify_intermittent: input vect is not numeric and could not be converted")
        if threshold=='':
            print("No threshold provided. Using vect[0] to compute scores with OFF threshold as percentage of threshold and excluding vect[0] from score computation for all OFF thesholds.")
            threshold = vect[0]
            vect = vect[1:len(vect)]
            print('Threshold:', threshold)
            
        ### Removing nan
        vect = vect[vect!=np.nan]

        ### Create low demand list names
        list_low_demand = ["zero", "perc_threshold"]
        for ind in ["floor_perc_quant_", "perc_quant_"]:
            list_low_demand.append(ind + str(quant).replace('0.', ''))

            for LD in list_low_demand:
                if LD=="zero":
                    low_demand = 0
                elif LD=="perc_threshold":
                    low_demand = perc*threshold
                elif LD=="floor_perc_quant_"+ str(quant).replace('0.', ''):
                    low_demand = max([0.250, 0.001*np.quantile(vect, quant)])    
                elif LD=="perc_quant_"+ str(quant).replace('0.', ''):
                    low_demand =  perc*np.quantile(vect, quant)
                
                nzd = vect[vect>low_demand]
                k = len(nzd)
                
                if sum(vect[vect>low_demand])>=2:
                    x = np.append([nzd[0]], [nzd[1:k] - nzd[0:(k-1)]])
                    
                    cv2 = Intermittent.cv2(nzd, highest, lowest)
                    adi = Intermittent.adi(x, highest, lowest)
                    sddi = Intermittent.sddi(x, highest, lowest)
                else:
                    cv2 = np.nan
                    adi = np.nan
                    sddi = np.nan
                
                res = pd.DataFrame.from_dict({'type': [LD], 'k': [k], 'low_demand': [low_demand], 'cv2': [cv2], 'adi': [adi], 'sddi': [sddi]})
        
        return res
    
    def enh_idclass5(vect, threshold, perc, quant, highest, lowest):   
        ''' Computes indicator values
        :params: vect as numpy array, threshold as numeric, perc as numeric, quant as numeric, highest and lowest as scalars 0<=x<=1 as winsorization percentages
        :return: a dictionary
        '''    
        if isinstance(vect,(np.ndarray))==False:
            try:
                vect = np.array(vect)
            except:        
                raise Exception("identify_intermittent: input vect is not numeric and could not be converted")
        if threshold=='':
            print("No threshold provided. Using vect[0] to compute scores with OFF threshold as percentage of threshold and excluding vect[0] from score computation for all OFF thesholds.")
            threshold = vect[0]
            vect = vect[1:len(vect)]
            print('Threshold:', threshold)
            
        ### Removing nan
        vect = vect[vect!=np.nan]

        ### Z function
        def Z(quant):
            cond1 = max([perc * np.quantile(vect, quant), 0.1*perc*np.quantile(vect, quant), 0.25])
            cond2 = min([perc * np.quantile(vect, quant), 0.1*perc*np.quantile(vect, quant)])
            if 0.25 >= cond1:
                return 0.25
            elif 0.25<cond2:
                return 0.1*perc*np.quantile(vect, quant)
            else:
                return perc*np.quantile(vect, quant)
            
        ### Low demand
        low_demand_name = "mix_floor_Q_" + str(quant).replace('0.', '')
        dict_low_demand = {low_demand_name: {'low_demand': Z(quant)}}

        for LD in list(dict_low_demand.keys()):
            low_demand = dict_low_demand[LD]['low_demand']
            nzd = vect[vect>low_demand]
            k = len(nzd)
            
            if sum(vect[vect>low_demand])>=2:
                x = np.array([nzd[0]]) + [nzd[1:k] - nzd[0:(k-1)]] + np.array([len(vect)+1-nzd[k-1]])

                cv2 = Intermittent.cv2(nzd, highest, lowest)
                adi = Intermittent.adi(x, highest, lowest)
                sddi = Intermittent.sddi(x, highest, lowest)
            else:
                cv2 = np.nan
                adi = np.nan
                sddi = np.nan

            res = pd.DataFrame.from_dict({'type': [LD], 'k': [k], 'low_demand': [low_demand], 'cv2': [cv2], 'adi': [adi], 'sddi': [sddi]})
            
        return res
    
    def classify_intermittent(df, type, thres_cv2_constant, thres_cv2, thres_adi, thres_sddi, min_time_cons):
        ''' Classifies intermittent time series based on indicator values
        :params: df as pandas dataframe, type as string, thres_cv2_constant as numeric, thres_cv2 as numeric, thres_adi as numeric, thres_sddi as numeric, min_time_cons as numeric
        :return: a pandas dataframe
        '''
        # Excluding the ids for which the indicators are np.nan
        score_no_nan = df.dropna()

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

