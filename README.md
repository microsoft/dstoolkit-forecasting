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

![Alt text](Docs/Images/intermkittent_ts.png?raw=true "Intermittent time series")

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
- Linear regression (implemented)
- Gradient boosting (implemented)
- Random Forest (implemented)
- ARIMA
- Prophet

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

