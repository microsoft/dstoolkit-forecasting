from Code.Scoring.scoring import Scoring
from Code.Scoring.train import Training
from Code.Scoring.forecast import Forecasting
from Code.Scoring.kpi import Kpi

class Trainer():
    def __init__(self, algorithms) -> None:
        self.algorithms = algorithms
        self.trained_model = None


    def add_algorithm(self, algorithm):
        self.algorithms.append(algorithm)

    def train(self, train_ds, test_ds):
        best_algorithm = Scoring.find_best_algorithm(train_ds.y_col, train_ds, test_ds, self.algorithms)
        self.trained_model = Training.train(train_ds, self.algorithms[best_algorithm])

        return self.trained_model, best_algorithm

    def forcast(self, forcast_ds):
        forecasted_model = Forecasting.forecast(forcast_ds, self.trained_model)
        return forecasted_model['df_fcst']

    def forcast_to_pbi_csv(self, test_ds, filename = "forcast.csv"):
        self.forcasted_df = self.forcast(test_ds).to_csv(filename, index=False)
        return self.forcasted_df

    def compute_model_kpi(self, df_pbi):
        df_pbi.loc[:, 'error'] = df_pbi['fcst'] - df_pbi[y]
        df_pbi.loc[:, 'absolute_error'] = abs(df_pbi['fcst'] - df_pbi[y])
        df_pbi.loc[:, 'absolute_percentage_error'] = abs(df_pbi['fcst'] - df_pbi[y])/df_pbi[y]

        print("MAE:", round(df_pbi.loc[:, 'absolute_error'].mean(), 0))
        print("MAPE:", round(df_pbi.loc[:, 'absolute_percentage_error'].mean(), 2))
