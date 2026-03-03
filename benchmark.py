import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import math
import time
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn import model_selection
import optuna
import warnings

warnings.filterwarnings('ignore')

def model_performance(y_test, y_pred):
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

'''Benchmark models: LR, SVR, MLP, XGB, LGB, RF, ET, CB'''
def _benchmark_lr_(train, test, target):
    start = time.time()
    regressor = LinearRegression()
    regressor.fit(train.drop(target, axis=1), train[target])
    predictions = regressor.predict(test.drop(target, axis=1))
    performance = model_performance(test[target].values, predictions)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'LinearRegression'
    performance['Training time'] = running_time
    return performance

def _benchmark_svr_(train, test, target):
    start = time.time()
    regressor = SVR()
    regressor.fit(train.drop(target, axis=1), train[target])
    predictions = regressor.predict(test.drop(target, axis=1))
    performance = model_performance(test[target].values, predictions)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'Support Vector Regression'
    performance['Training time'] = running_time
    return performance

def _benchmark_mlp_(train, test, target):
    start = time.time()
    regressor = MLPRegressor()
    regressor.fit(train.drop(target, axis=1), train[target])
    predictions = regressor.predict(test.drop(target, axis=1))
    performance = model_performance(test[target].values, predictions)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'Multilayer Perception'
    performance['Training time'] = running_time
    return performance

def _benchmark_xgb_(train, test, target):
    start = time.time()
    regressor = xgb.XGBRegressor(verbosity=0)
    regressor.fit(train.drop(target, axis=1), train[target])
    predictions = regressor.predict(test.drop(target, axis=1))
    performance = model_performance(test[target].values, predictions)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'XGBoost'
    performance['Training time'] = running_time
    return performance

def _benchmark_lgb_(train, test, target):
    start = time.time()
    regressor = lgb.LGBMRegressor()
    regressor.fit(train.drop(target, axis=1), train[target])
    predictions = regressor.predict(test.drop(target, axis=1))
    performance = model_performance(test[target].values, predictions)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'LightGBM'
    performance['Training time'] = running_time
    return performance

def _benchmark_rf_(train, test, target):
    start = time.time()
    regressor = RandomForestRegressor()
    regressor.fit(train.drop(target, axis=1), train[target])

    print(model_performance(train[target].values,
                            regressor.predict(train.drop(target, axis=1))))
    predictions = regressor.predict(test.drop(target, axis=1))
    performance = model_performance(test[target].values, predictions)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'RandomForest'
    performance['Training time'] = running_time
    return performance

def _benchmark_extra_(train, test, target):
    start = time.time()
    regressor = ExtraTreesRegressor()
    regressor.fit(train.drop(target, axis=1), train[target])
    predictions = regressor.predict(test.drop(target, axis=1))
    performance = model_performance(test[target].values, predictions)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'ExtraTree'
    performance['Training time'] = running_time
    return performance

def _benchmark_cat_(train, test, target):
    start = time.time()
    regressor = CatBoostRegressor(verbose=False)
    regressor.fit(train.drop(target, axis=1), train[target])
    predictions = regressor.predict(test.drop(target, axis=1))
    performance = model_performance(test[target].values, predictions)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))
    performance['Framework'] = 'CatBoost'
    performance['Training time'] = running_time
    return performance

'''Benchmark optuna-models: LR, SVR, MLP, XGB, LGB, RF, ET, CB'''
def fitted_models(train, test, target, train_method):
    if train_method == 'LR':
        regressor = LinearRegression()
        best_params = None

    # SVR
    elif train_method == 'SVR':
        def objective(trial):
            params={
            'epsilon' : trial.suggest_float('epsilon', 1e-4, 1.0, log=True),
            'C' : trial.suggest_float('C', 0.01, 1e5, log=True),
            'kernel' : trial.suggest_categorical('kernel', ['poly', 'rbf', 'linear', 'sigmoid']),
            'degree' :  trial.suggest_int("degree", 1, 4),
            'tol':1e-3,
            'max_iter':5000,
            }
            model_obj = SVR(**params)
            score = model_selection.cross_val_score(model_obj,
                                                    train.drop(target, axis=1),
                                                    train[target],
                                                    n_jobs=-1,
                                                    cv=3,
                                                    scoring='r2')
            accuracy = score.mean()
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        regressor = SVR(**best_params, tol=1e-3,max_iter=5000)

    # MLP
    elif train_method == 'MLP':
        def objective(trial):
            params={
                'hidden_layer_sizes':(trial.suggest_int('n_neurons_layer1', 10, 100),
                                      trial.suggest_int('n_neurons_layer2', 10, 100),
                                      trial.suggest_int('n_neurons_layer3', 10, 100)),
                'activation' : trial.suggest_categorical("activation", ['tanh', 'relu']),
                'alpha' : trial.suggest_float("alpha", 1e-7, 1e-1),
                'learning_rate_init' : trial.suggest_float("learning_rate_init", 1e-4, 1e-1),
                'learning_rate' : trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
                'tol' : 1e-3,
                'max_iter':1000
            }
            model_obj = MLPRegressor(**params)
            score = model_selection.cross_val_score(model_obj,
                                                    train.drop(target, axis=1),
                                                    train[target],
                                                    n_jobs=-1,
                                                    cv=3,
                                                    scoring='r2')
            accuracy = score.mean()
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        hidden_layer_sizes = (best_params['n_neurons_layer1'],
                              best_params['n_neurons_layer2'],
                              best_params['n_neurons_layer3'])
        del best_params['n_neurons_layer1']
        del best_params['n_neurons_layer2']
        del best_params['n_neurons_layer3']
        regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                 tol=1e-3,
                                 max_iter=1000,
                                 **best_params)

    # XGBoost
    elif train_method == 'XGB':
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            }
            model_obj=xgb.XGBRegressor(**params)
            score = model_selection.cross_val_score(model_obj,
                                                    train.drop(target, axis=1),
                                                    train[target],
                                                    n_jobs=-1,
                                                    cv=3,
                                                    scoring='r2')
            accuracy = score.mean()
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        regressor = xgb.XGBRegressor(**best_params)

    elif train_method == 'LGB':
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int("num_leaves", 2, 100),
                'max_depth':trial.suggest_int("max_depth", 5, 20),
                'n_estimators': trial.suggest_int("n_estimators", 10, 100),
                'subsample': trial.suggest_uniform('subsample', 0.1, 1.0)
            }
            model_obj=lgb.LGBMRegressor(**params)
            score = model_selection.cross_val_score(model_obj,
                                                    train.drop(target, axis=1),
                                                    train[target],
                                                    n_jobs=-1,
                                                    cv=3,
                                                    scoring='r2')
            accuracy = score.mean()
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        regressor = lgb.LGBMRegressor(**best_params)

    # Random Forest Regression
    elif train_method == 'RF':
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 10, 100),
                'max_depth': trial.suggest_int("max_depth", 5, 20),
                'max_features':trial.suggest_uniform("max_features", 0.05, 1.0),
            }
            model_obj=RandomForestRegressor(**params)
            score = model_selection.cross_val_score(model_obj,
                                                    train.drop(target, axis=1),
                                                    train[target],
                                                    n_jobs=-1,
                                                    cv=3,
                                                    scoring='r2')
            accuracy = score.mean()
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        regressor = RandomForestRegressor(**best_params)

    # ExTra Tree regression
    elif train_method == 'ET':
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 10, 100),
                'max_features':trial.suggest_uniform("max_features", 0.05, 1.0),
            }
            model_obj=ExtraTreesRegressor(**params)
            score = model_selection.cross_val_score(model_obj,
                                                    train.drop(target, axis=1),
                                                    train[target],
                                                    n_jobs=-1,
                                                    cv=3,
                                                    scoring='r2')
            accuracy = score.mean()
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        regressor = ExtraTreesRegressor(**best_params)

    # CatBoost Regression
    elif train_method == 'CB':
        def objective(trial):
            params = {
                'iterations': 20,
                'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int("max_depth", 5, 16)
            }
            model_obj=CatBoostRegressor(**params)
            score = model_selection.cross_val_score(model_obj,
                                                    train.drop(target, axis=1),
                                                    train[target],
                                                    n_jobs=-1,
                                                    cv=3,
                                                    scoring='r2')
            accuracy = score.mean()
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        regressor = CatBoostRegressor(**best_params, iterations=100)
    return best_params, regressor
    # regressor.fit(train.drop(target, axis=1), train[target])
    # predictions = regressor.predict(test.drop(target, axis=1))
    # performance = model_performance(test[target].values, predictions)
    # finish = time.time()
    # running_time = ("%.2f" % ((finish - start) / 60))
    # # performance['Framework'] = 'LinearRegression'
    # performance['Training time'] = running_time
    # return performance