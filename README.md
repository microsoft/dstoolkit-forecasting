![banner](Docs/Images/banner.jpg)

# Forecasting 2.0 Accelerator

# Overview
This accelerator provides code and guidance to produce time series forecasting and time series profiling. The aim of this accelerator is to help data scientists to forecast multiple time series by building models based on the time-series profiling, by performing an accurate data preparation and by training and forecasting multiple time series based with models created ad-hoc for each profile. 

Time series modelling is defined as the combination of:
1. Choice of explanatory variables or regressors - which variables helps me in explaining the data generation process I want to forecast?
2. Choice of forecasting algorithm - which algorithm do I use to produce my forecast? Arima, linear regression, boosting model?
3. Choice of train set - how many observation do I use to train my model and produce my forecast?

Each model is optimized to better fit the data generation process of the phenomenon we want to forecast: from energy consumption to spare parts demand. Classification of time series in terms of profile, or consumption profile if we are referring to energy consumption, helps in defining the best fitting model in terms of choice of regressors (calendar variables or temperatures), forecasting algorithm (ARIMA vs exponential smoothing) and train set (one year or just few days of data). 

# Profiling (clustering) Time Series:​
The **goal** is to identify consumption patterns that are similar to each other in order to assign the optimal model in terms of min of MAE or MSE​. 

The **first step** is to identify those series that are classified as “intermittent” with respect to those “regular”​ and **then** proceed to perform a k-means cluster analysis only on the latter. 

The **expected output** is to label each time series as intermittent with respect to regular.

### Identifying intermittent time series:​

Definition of intermittent time series: intermittent time series or demand comes about when a product or a time series experiences several periods of zero demand. Often in these situation, when demand occurs it is small, and sometimes highly variable in size​

#### How to identify intermittent time series:​
Compute the following indicators such as
1. ​Average Inter-demand Interval (ADI), this parameter is period based which is calculated as average interval time between two demand occurrences​
2. Coefficient of Variation Squared (CV2), this statistical parameter is calculated as standard deviation of the *For correspondence demand divided by the average demand for non-zero demand periods. The squared coefficient of variation represents variability of demand size.​
3. Standard Deviation of Inter-demand Interval (SDDI) ​

Based on their values, it is possible to identify intermittent time series as:
- intermittent
- lumpy
- erratic
- unforecastable in terms of time volatility
- unforecastable in terms of quantity volatility
- regular time series ​

![Alt text](Docs/Images/intermittent_TS.png?raw=true "Intermittent time series")

#### Methods to forecast intermittent time series (not yet implemented in this accelerator):​
- Croston’s method​
- Adjusted Croston methods ​
- Model-based forecasting methods 
  - ARMA models​
  - DARMA models -> Discrete ARMA​
  - INARMA models -> Integer-valued ARMA (INARMA)​
- Alternative methods
  - Boostrapping​
  - Temporal aggregation​

### Clustering profiles​
  - Clustering regular time series using K-Means flat
  - Choose the optimal number of clusters ​
    - As a method to choose the optimal number of cluster, use max explained variance at the minimum number of cluster -> Elbow Method​
    ![Alt text](Docs/Images/elbow.png?raw=true "Elbow method")
    - Check weather identified profiles have a business meaning
    - Define and assign a best model:
      - use temperatures if thermal consumption is present in an energy consumption use case 
        ![Alt text](Docs/Images/thermal.png?raw=true "Thermal time series")
      - use calenadar variables correlation with temperatures is not present 
        ![Alt text](Docs/Images/calendar.png?raw=true "Calendar time series")

#### Methods to forecast regular time series
|#    | Model | Library   | Status | Notes |
| :---        |    :----:   |          ---: |  ---: |---: |
| 1 | Linear regression      | [statsmodel](https://www.statsmodels.org/stable/api.html#univariate-time-series-analysis)     |Implemented    |  |
| 2 | Gradient boosting      | [xgboost](https://xgboost.readthedocs.io/en/stable/)     |Implemented    |  |
| 3 | Random forest      | [statsmodel](https://www.statsmodels.org/stable/api.html#univariate-time-series-analysis)     |Implemented    |  |
| 4 | Kats |[Kats](https://facebookresearch.github.io/Kats/api/) |Not yet tmplemented    |  |
| 5 | Prophet | [Prophet](https://facebook.github.io/prophet/docs/quick_start.html#python-api)|Not yet implemented    |Decompose into trend + season + holiday, etc |
| 6 |Neural networks|[Neural prophet](https://neuralprophet.com/html/index.html) |Not yet implemented    |  |
| 7 |Probabilistic model|[PyFlux](https://github.com/RJT1990/pyflux) |Not yet implemented    |  |
| 8|Scikit-learn wrapper|[Sktime](https://www.sktime.org/en/stable/) |Not yet implemented    |  |
| 9|Automatic time series|[AutoTimeSeries](https://github.com/AutoViML/Auto_TS) |Not yet implemented    |  |
| 10 |Create synthetic time series for model testing|[TimeSynth](https://github.com/TimeSynth/TimeSynth) |Not yet implemented    |  |
| 11 |Computes series characteristics|[Tsfresh](https://github.com/blue-yonder/tsfresh) |Not yet implemented    |  |
| 12 |ARIMA and deep NN|[Darts](https://github.com/unit8co/darts) |Not yet implemented    |  |
| 13 |Uber forecasting package|[Orbit](https://github.com/uber/orbit) |Not yet implemented    | pystan backend |
| 14 |Converting dates|[Arrow](https://github.com/pastas/pastas) |Not yet implemented    |  |
| 15 |Hydro(geo)logical time series analysis|[Pastas](https://github.com/pastas/pastas) |Not yet implemented    |  |
| 16|Deep learning|[Flow forecast](https://github.com/AIStream-Peelout/flow-forecast) |Not yet implemented    |  |
| 17 |Automating iterative tasks of machine learning model development|[AutoML in Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#supported-models) |Not yet implemented    |  |
| 18 |Netflix forecasting package | [Metaflow](https://docs.metaflow.org/introduction/what-is-metaflow) |Not yet implemented    |  |


## References: ​
Lancaster Centre For Marketing Analytics and Forecasting ​(https://www.lancaster.ac.uk/lums/research/areas-of-expertise/centre-for-marketing-analytics-and-forecasting/)

Methods for Intermittent Demand Forecasting​ (https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf)

# Getting Started
1. Create a new conda environment named forecasting_energy using the forecasting_energy_env.yml file in the Environment folder in the repository. To install a new environment using conda, you can access Anaconda navigator, click on import, name the new environment as forecasting_energy, select Python 3.8 and use the path to forecasting_energy_env.yml to install the new environment. 
2. To have an idea of software dependencies, read Requirements.txt
3. Create a config.yaml in Configuration folder, in order to run the code on your local machine/virtual machine. This is an example of the file:

### config.yaml file example

data_dir:
  input_path: "Data/Input"
  output_path: "Data/Output"
  plot_path: "Data/Plots"

4. Create your input, output and plot path 
5. Load the test dataset from Kaggle (https://www.kaggle.com/arashnic/building-sites-power-consumption-dataset/download"), unzip it and save it in your input folder

## Default Directory Structure

```bash
├───Code     # Folder containing all the custom function created for this accelerator
│   ├───Plotting    # Plotting functions
│   └───Profiling   # Profiling time series functions
│       ├───Intermittent # Identification and classification of intermittent time series functions
│   └───Regressors # Create useful time series regressors, such as calendar variables or temperature transformations
│   └───Scoring # Create train and test sets, training, forecasting and computing kpis functions
│   └───Utils   # Several utils functions recalled in the notebooks
├── Configuration # config.py that lead to config.yaml. with configuration 
├───Docs # Additional documents
├───Notebooks     # Notebooks to do Data Preparation, Scoring and Forecasting and Profiling 
├── .gitignore
├── CODE_OF_CONDUCT.md
├── LICENSE.md
├── README.md
|── requirements.txt
├── SECURITY.md
└── SUPPORT.md
```

## Build and Test
1. Create a config.yaml as described above and compile it as:
    - In data_dir set your folder tree for input, output and plot folder
    - In saving choose your saving preferences

## Notebooks
Notebooks are available in the Notebooks folder and provide guidance to use the Forecast 2.0 functions. 
### 1. Data Exploration
A notebook that provides an exploratory data analysis in order to understand the type of time series we are dealing with
### 2. EnergyPredictionDataPreparation
A notebook that helps with Time Series Data Preparation, in particular how to deal with NAs, how to aggregate time series and how to add create useful regressors (e.g. calendar variables)
### 3. ProfilingIntermittent
A notebook that profiles time series by classify them among regular, intermittent, lumpy, erratic, unforecastable in terms of time, unforecastable in terms of quantity
### 4. ClusteringRegular
A notebook that performs a k-means flat cluster analysis on those time series that were classified as regular
### 5. EnergyPredictionScoring
A notebook that helps you produce a forecast, plotting the results and compute KPIs on a panel dataframe, where you have multiple timeseries identified by a given id (e.g. multiple sensors time series, multiple plants or site-id energy consumption, etc)

# Functions
Functions are available in the Code folder.

## Plotting
### Class Plots
```bash
sliding_line_plot(df, serie_to_plot, id, i, chart_title="")
```
Creates a sliding time series chart

```bash
sliding_fcst_plot(df, predict_col, expected_values, chart_title="", kpi=True)
```
Creates a forecast vs actual sliding time series chart, with KPI option

## Regressors
### Class Regressors
```bash  
create_interactions(df, var1, var2)
```
Adds interaction terms between two variables as var1*var2 to dataframe

```bash  
create_non_linear_terms(df, var, n)
```
Adds non linear terms as var^2 to dataframe

```bash  
add_holidays_by_country(df, date_var, country)
```
Adds holidays a dummy variable (0/1) to dataframe

```bash  
add_weekdays(df, date_var)
```
Adds weekdays a dummy variables (0/1) for each weekday to dataframe

```bash  
add_months(df, date_var)
```
Adds months a dummy variables (0/1) for each month to dataframe

```bash  
calculate_degree_days(df, base_temperature, temperature)
```
Calculate the Degree Days Heating and Cooling values

```bash  
merge_holidays_by_date(df, df_holidays, id)
```
Merge Holiday df with the train df

```bash  
merge_additional_days_off(df, df_metadata, id, dict_days_off)
```
Merge Site Weekend data with train df

```bash  
merge_weather(df, weather, date_var, id)
```
Merge weather data into the train df

### Class SimilarDay:
```bash
get_similar_days_in_previous_year(dates, country)
```
Retrieves the similar day for a given date

```bash
get_similar_days_in_previous_week(dates, country)
```
Retrieves the similar day for a given date

```bash
get_similar_day_in_previous_year(d, holiday_calendar)
```
Retrieves the similar day for a given date. If the given date is not an holiday, the similar day is the closest day of the previous year in terms of calendar position which shares the weekday. If such a date is an holiday, the same weekday of the week before is considered. 

```bash
get_similar_day_in_previous_week(d, holiday_calendar)
```
Retrieves the similar day for a given date. If the given date is not an holiday, the similar day is the closest day of the previous year in terms of calendar position which shares the weekday. If such a date is an holiday, the same weekday of the week before is considered. If the given date is an holiday, its similar day is the closest holiday to the given date in the previous year.

### Class StandardConsumption:   
```bash
get_standard_consumption_as_mean(df, id, date_var, var, country)
```
Retrieves the standard consumption for a given date as hourly monthly mean differentiated by holiday, weekend, weekdays

### Class Temperatures:
```bash       
ten_year(df, id, date_var = 'date_daily', start_date ='', end_date='31/12/2050')
```
Computes ten year averages temperatures and As-Is temperatures: where available use actual temp, if not use ten year averages

```bash 
get_minimum_consumption(df, date_var, var, country)
```
Retrieves the minimum consumption for a given date as hourly monthly minimum value differentiated by holiday, weekend, night

## Scoring
### Class Training
```bash 
train(dict_model_to_train, model)
```
Generate train

### Class Forecasting
```bash
forecast(dict_test, trained_model)
```
Generate forecast

### Class Scoring
```bash 
find_best_algorithm(y, dict_train, dict_test, dict_algorithms, out_of_sample)
```
Finds the best performing algorithm in terms of min mean absolute error

```bash 
stats_per_site(df, id, date_var)
```
Helper function to identify amount of data per site

```bash 
resample_train_data(df, date_var, id, predict_col, sampling="D")
```
Resample the data to a particular frequency

### Class TrainTest
```bash 
define_train_test_set_dates(df, y, train_start_date, train_end_date, test_start_date, test_end_date, test_size=0.33)
```
Defines train and test dates if left blank  

```bash 
def_train(df, y, list_id, train_start_date='', train_end_date='')
```
Define train dataset 

```bash 
def_test(df, y, list_id, test_start_date='', test_end_date='')
```
Define test dataset

## Kpi
### Class Kpi
```bash
find_mae(y, dict_train, dict_test, dict_models):
```
Compute mean absolute error
```bash
compute_error(df, fcst, y):
```    
Compute error as forecast-actual
```bash
compute_absolute_error(df, fcst, y):
```    
Compute absolute error as abs(forecast-actual)
```bash
compute_absolute_percentage_error(df, fcst, y):
```     
Compute absolute % error
```bash
compute_mean_error(df, fcst, y):
```      
Compute mean  error
```bash
compute_mae(df, fcst, y):
```   
Compute mean absolute error
```bash
compute_mape(df, fcst, y):
```     
Compute mean absolute % error

## Utils
### Class Utils
```bash
def camel_to_snake(name)
```
Changes string from camel case to snake case
```bash
columns_camel_to_snake(df)
```
Changes dataframe columns from camel case to snake case
```bash
find_date(df)
```
Finds date columns in a dataframe
```bash
find_match_in_list(list_to_match, match_to_find):
```
Finds a match in a list given a list of possible words to match
```bash
delta_format(delta: np.timedelta64) -> str:
```
Identifies frequency in numpy timedelta
```bash
find_freq(timedelta):
```
Finds frequency in numpy timedelta
```bash
find_freq_in_dataframe(df, date_var)
```
Finds frequency in pandas dataframe
```bash
get_project_root(Path):
```
Finds the parent folder of the parent folder 
```bash
add_daily_date(df):
```
Adds a date variable at daily frequency to dataframe
```bash
find_categorical_variables(df):
```
Finds categorical variables in pandas dataframe
```bash
resample_data(df, id, date_var, sampling, dict_grouping)
```
Resample by aggregating the data to a particular frequency as defined in dict_grouping as{variable_to_resample: 'function_to_apply'}, i.e.{value: 'sum'}
```bash
add_seq(df, date_var, serie, freq, end_date='', start_date='')
```
Creates a sequence of completes date/hours to a dataframe
```bash
check_length_time_serie(df, date_var, index)
```
Checks the length that a time sequence of completes date/hours should have, so that it can be compared 
with actual observation
```bash
match_to_find(serie_to_find)
```
Finds a match in a list of possible words to match
```bash
find_match(df, serie_name, match_to_find):
```
Finds a match in a dataframe serie given a list of possible words to match
```bash
find_match_in_list(list_to_match, match_to_find)
```
Finds a match in a list given a list of possible words to match
```bash
id_outliers_IQR(df, q1, q3, date_var, id, var, freq_var)
```
Identifies outliers creatinga dummy variable (0/1) called outlier using IQR method, where quantile value can be set
        
### Class AlphabeticalCombinations
```bash
write_neat_csv(saving_file, df_fcst)
```
Writes neat csv
```bash        
convert(string)
```
Convert string to list
```bash
excel_columns()
```
Counts excel columns
```bash
write_beautiful_excel(saving_file, dict_df_to_write)
```
Writes beautiful excel
```bash
write_beautiful_excel_table(saving_file, dict_df_to_write)
```
Writes beautiful excel tables

# Contributing
This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


