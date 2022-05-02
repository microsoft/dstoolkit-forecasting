# data elaboration functions
import pandas as pd
import numpy as np
import re

# file management functions
import os
import glob

# time management functions
import datetime as dt

# plot functions
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# custom functions
from Configuration.config import cfg_path
from Code.Utils.utils import Utils

class Plots:
    def get_text_positions(x_data, y_data, txt_width, txt_height):
        a = zip(y_data, x_data)
        text_positions = y_data.copy()
        for index, (y, x) in enumerate(a):
            local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                                and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
            if local_text_positions:
                sorted_ltp = sorted(local_text_positions)
                if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                    differ = np.diff(sorted_ltp, axis=0)
                    a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                    text_positions[index] = sorted_ltp[-1][0] + txt_height
                    for k, (j, m) in enumerate(differ):
                        #j is the vertical distance between words
                        if j > txt_height * 2: #if True then room to fit a word in
                            a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                            text_positions[index] = sorted_ltp[k][0] + txt_height
                            break
        return text_positions

    def text_plotter(x_data, y_data, text_positions, axis,txt_width,txt_height):
        for x,y,t in zip(x_data, y_data, text_positions):
            axis.text(x - txt_width, 1.01*t, '%d'%int(y),rotation=0, color='blue')
            if y != t:
                axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1, 
                        head_width=txt_width, head_length=txt_height*0.5, 
                        zorder=0,length_includes_head=True)
                
    def sliding_line_plot(df, serie_to_plot, id, i, chart_title=""):
        """
        Creates a time series plot with sliding dates
        :params: df as pandas dataframe
        :return: html file with plot
        """
        
        ### Setup          
        date = Utils.find_date(df)
        
        ## Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(df.loc[df[id] == i, date]), y=list(df.loc[df[id] == i, serie_to_plot]), name=str(i)))

        # Set title
        if chart_title!="":
            fig.update_layout(
                title_text=chart_title
            )

        else:
            chart_title = serie_to_plot.capitalize() + ' ' + str(id) + ' ' + str(i)
            fig.update_layout(
                title_text=chart_title
            )            
        
        print('sliding_line_plot: plotting', chart_title)

        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                                label="1m",
                                step="month",
                                stepmode="backward"),
                        dict(count=3,
                                label="3m",
                                step="month",
                                stepmode="backward"),
                        dict(count=6,
                                label="6m",
                                step="month",
                                stepmode="backward"),
                        dict(count=1,
                                label="YTD",
                                step="year",
                                stepmode="todate"),
                        dict(count=1,
                                label="1y",
                                step="year",
                                stepmode="backward"),
                        dict(step="all")
                        ])
                    ),
                rangeslider=dict(
                    visible=True
                    ),
                type="date"
                )
        )
        return fig
    
    
    def sliding_fcst_plot(df, predict_col, expected_values, chart_title="", kpi=True):
        """
        Creates a time series plot with sliding dates
        :params: df as pandas dataframe, chart_title as string, kpi as boolean 
        :return: html file with plot
        """
        
        ### Setup
        date = Utils.find_date(df)
        if isinstance(date, list):
            date = list(set(Utils.find_date(df)) - set(['train_start_date', 'train_end_date', 'test_start_date', 'test_end_date']))[0]
        
        y = predict_col
        fcst = expected_values
        
        ## Adding model info to chart title
        if 'best_model' in list(df.columns):
            model = df['best_model'].unique()[0]
            chart_title = str(chart_title) + ' - ' + model 
        else:
            chart_title = str(chart_title)     
        
        ## Checking KPI
        if kpi == True:
            try:
                mape = str(round(df.loc[~df.absolute_percentage_error.isnull(), 'absolute_percentage_error'].mean()*100, 2))
                min_mape_date = min(df.loc[~df.absolute_percentage_error.isnull(), date]).strftime("%d-%m-%Y")
                max_mape_date = max(df.loc[~df.absolute_percentage_error.isnull(), date]).strftime("%d-%m-%Y")
                chart_title = chart_title + ' - MAPE: ' +  mape + "% from " + min_mape_date + ' to ' + max_mape_date
            except:
                chart_title = str(chart_title)
        else:
            chart_title = str(chart_title)    

        ## Create figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=list(df[date]), y=list(df[y]), name=y))
        fig.add_trace(go.Scatter(x=list(df[date]), y=list(df[fcst]), name=fcst))

        # Set title
        if chart_title!="":
            fig.update_layout(
                title_text=chart_title
            )
        else:
            fig.update_layout(
                title_text="Forecasting " + y.capitalize()
            )     
            
        # Add annotations
        for col in ['train_start_date', 'train_end_date', 'test_start_date', 'test_end_date']: 
            if col in list(df.columns):
                col_date = pd.to_datetime(str(df[col].unique()[0])).strftime('%Y-%m-%d')
                date_value = df[col].unique()[0]
                unique_index = pd.Index(list(df[date].unique()))
                closest_date = df.loc[unique_index.get_loc(date_value,method='nearest'), date]
                x_value = pd.to_datetime(df.loc[df[date]==closest_date, date].reset_index(drop=True)[0], format='%Y-%m-%d') 
                y_value = pd.to_numeric(df.loc[df[date]==closest_date, y].reset_index(drop=True)[0])
                fig.add_annotation(
                x=x_value, 
                y=y_value,
                #textangle=45,
                text= col + ': ' +  str(col_date),
                showarrow=True,
                arrowhead=1, 
                arrowsize=1,
                arrowwidth=2,
                font = dict(
                color="black",
                size=16
            ))
            else:
                print('No annotation available for', col)

        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                                label="1m",
                                step="month",
                                stepmode="backward"),
                        dict(count=3,
                                label="3m",
                                step="month",
                                stepmode="backward"),
                        dict(count=6,
                                label="6m",
                                step="month",
                                stepmode="backward"),
                        dict(count=1,
                                label="YTD",
                                step="year",
                                stepmode="todate"),
                        dict(count=1,
                                label="1y",
                                step="year",
                                stepmode="backward"),
                        dict(step="all")
                        ])
                    ),
                rangeslider=dict(
                    visible=True
                    ),
                type="date"
                )
        )
            
        return fig

   