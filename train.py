import os
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import math
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from benchmark import fitted_models
import joblib

'''
Machine learning fit
'''
def data_split(data,
              target,
              train_size=0.7,
              random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(data.drop(target, axis=1),
                                                        data[target],
                                                        train_size=train_size,
                                                        random_state=random_state)
    train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
    test = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)
    return train, test

def model_performance(y_test, y_pred):
    # ML perspective
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    performance_result = {'RMSE': math.sqrt(mse),
                          'NRMSE': (((math.sqrt(mse)) / abs(np.mean(y_test))) * 100),
                          'MAE': mae,
                          'MAPE': mean_absolute_percentage_error(y_test, y_pred),
                          'R2': r2,
                          'std': y_pred.std(),
                          'rho': pearsonr(y_pred, y_test)[0],
                          'ref': y_test.std()
                          }
    return performance_result

'''
Machine learning model fit
'''
class Ml_model:
    def __init__(self, df, target, train_size):
        self.df = df
        self.target = target
        self.train_size = train_size
        train, test = data_split(self.df, target=self.target, train_size=self.train_size)
        self.train = train
        self.test = test

    def fit_predict(self, train_method, xdata=None):
        best_params, regressor = fitted_models(self.train, self.test, self.target, train_method=train_method)
        start = time.time()
        regressor.fit(self.train.drop(self.target, axis=1), self.train[self.target])
        finish = time.time()
        training_time = finish - start

        # save the model as pkl
        os.makedirs('saved_models', exist_ok=True)
        joblib.dump(regressor, f'./saved_models/{train_method}.pkl')

        train_xdata = self.train.drop(self.target, axis=1)
        test_xdata = self.test.drop(self.target, axis=1)
        training_performance = model_performance(self.train['Compressive strength'], regressor.predict(train_xdata))
        test_performance=model_performance(self.test['Compressive strength'], regressor.predict(test_xdata))
        # return regressor.predict(xdata)

        return best_params, training_time, training_performance, test_performance

    def _performance(self):
        train_ps = []
        test_ps = []
        # 'LR', 'SVR', 'MLP', 'XGB', 'LGB', 'RF', 'ET', 'CB'
        for train_method in ['LR', 'SVR', 'MLP', 'XGB', 'LGB', 'RF', 'ET', 'CB']:
            best_params, training_time, train_p, test_p =self.fit_predict(train_method=train_method)

            train_p['Method'] = train_method
            train_p['Data type'] = 'Train set'
            train_p['Training time'] = training_time
            train_p['Best params'] = best_params

            test_p['Method'] = train_method
            test_p['Data type'] = 'Test set'
            train_ps.append(train_p)
            test_ps.append(test_p)

        train_ps = pd.DataFrame(train_ps)
        test_ps = pd.DataFrame(test_ps)

        # train_ps.to_csv('./performance_n/train_p.csv', index=False)
        # test_ps.to_csv('./performance_n/test_p.csv', index=False)
        max_row = test_ps.loc[test_ps['R2'].idxmax()]
        best_method = max_row['Method']
        best_model = joblib.load(f'./saved_models/{best_method}.pkl')
        return pd.concat([train_ps, test_ps], axis=0).reset_index(drop=True), best_model


if __name__ == "__main__":
    df = Ml_model(pd.read_csv('data.csv'), target='Compressive strength', train_size=0.7)._performance()
    train_df = df[df['Data type'] == 'Train set']
    test_df = df[df['Data type'] == 'Test set']
    max_row = test_df.loc[test_df['R2'].idxmax()]
    print(max_row['Method'])
    print(train_df[train_df['Method'] == max_row['Method']].loc[0, 'Best params'])

