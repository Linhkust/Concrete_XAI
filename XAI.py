import pandas as pd
import numpy as np
from sklearn.inspection import partial_dependence, PartialDependenceDisplay, permutation_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap
from train import data_split
import statsmodels.api as sm


# 方法1: 使用线性插值找零点
def find_zeros_by_interpolation(x, y):
    """通过线性插值找到y=0的x值"""
    zeros = []

    # 遍历所有线段
    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]

        # 检查是否跨过零点
        if y1 == 0:
            zeros.append(x[i])
        elif y1 * y2 < 0:  # 异号，说明跨过零点
            # 线性插值
            t = -y1 / (y2 - y1)
            x_zero = x[i] + t * (x[i + 1] - x[i])
            zeros.append(x_zero)

    return np.array(zeros)

class Importance(object):
    def __init__(self, df, target):
        self.df = df
        self.target = target
        train, test = data_split(self.df, target=self.target)
        self.train = train
        self.test = test

    # partial dependence plot
    def pdp(self, model, features, df_type='train'):
        model.fit(self.train.drop(self.target, axis=1), self.train[self.target])
        feature_names = self.train.columns

        if df_type=='train':
            fig, ax = plt.subplots(1, 1)
            pdp_train = PartialDependenceDisplay.from_estimator(model,
                                                          self.train.drop(self.target, axis=1),
                                                          features=features,
                                                          kind='average',
                                                          ax=ax,
                                                          feature_names=feature_names,
                                                          n_jobs=-1)

            if len(features) == 1:
                pdp_train.axes_[0][0].set_ylabel("Compressive strength (MPa)")
                pdp_train.axes_[0][0].set_title("Partial dependence plot (train set)")
            else:
                pass
            return ax

        elif df_type=='test':
            fig, ax = plt.subplots(1, 1)
            pdp_test = PartialDependenceDisplay.from_estimator(model,
                                                               self.test.drop(self.target, axis=1),
                                                                features=features,
                                                                kind='average',
                                                                ax=ax,
                                                                feature_names=feature_names,
                                                                n_jobs=-1)
            if len(features[0]) == 1:
                pdp_test.axes_[0][0].set_ylabel("Compressive strength (MPa)")
                pdp_test.axes_[0][0].set_title("Partial dependence plot (test set)")
            return ax

        else:
            fig, ax = plt.subplots(1, 1)
            pdp_test = PartialDependenceDisplay.from_estimator(model,
                                                               self.df.drop(self.target, axis=1),
                                                               features=features,
                                                               kind='average',
                                                               ax=ax,
                                                               feature_names=feature_names,
                                                               n_jobs=-1)
            if type(features[0]) is str:
                pdp_test.axes_[0][0].set_ylabel("Compressive strength (MPa)")
            else:
                pass
            return ax

    # permutation importance analysis
    def pfi(self, model, df_type='all'):
        model.fit(self.train.drop(self.target, axis=1), self.train[self.target])
        if df_type == 'train':
            result = permutation_importance(model,
                                            self.train.drop(self.target, axis=1),
                                            self.train[self.target],
                                            scoring='r2',
                                            n_repeats=10,
                                            random_state=42,
                                            n_jobs=-1)

            sorted_importances_idx = result.importances_mean.argsort()
            importances = pd.DataFrame(
                result.importances[sorted_importances_idx].T,
                columns=self.train.drop(self.target, axis=1).columns[sorted_importances_idx])
            ax = importances.plot.box(vert=False, whis=10)
            ax.set_title("Permutation feature importance (train set)")
            ax.axvline(x=0, color="k", linestyle="--")
            ax.set_xlabel("Decrease in R squared")
            ax.figure.tight_layout()
            return ax

        elif df_type == 'test':
            result = permutation_importance(model,
                                            self.test.drop(self.target, axis=1),
                                            self.test[self.target],
                                            scoring='r2',
                                            n_repeats=10,
                                            random_state=42,
                                            n_jobs=-1)

            sorted_importances_idx = result.importances_mean.argsort()
            importances = pd.DataFrame(
                result.importances[sorted_importances_idx].T,
                columns=self.test.drop(self.target, axis=1).columns[sorted_importances_idx])
            ax = importances.plot.box(vert=False, whis=10)
            ax.set_title("Permutation feature importance (test set)")
            ax.axvline(x=0, color="k", linestyle="--")
            ax.set_xlabel("Decrease in R squared")
            ax.figure.tight_layout()
            return ax

        else:
            result = permutation_importance(model,
                                            self.df.drop(self.target, axis=1),
                                            self.df[self.target],
                                            scoring='r2',
                                            n_repeats=10,
                                            random_state=42,
                                            n_jobs=-1)

            sorted_importances_idx = result.importances_mean.argsort()
            importances = pd.DataFrame(
                result.importances[sorted_importances_idx].T,
                columns=self.df.drop(self.target, axis=1).columns[sorted_importances_idx])
            ax = importances.plot.box(vert=False, whis=10)
            ax.set_title("Permutation feature importance")
            ax.axvline(x=0, color="k", linestyle="--")
            ax.set_xlabel("Decrease in R squared")
            ax.figure.tight_layout()
            return ax


    # SHAP feature importance and dependence plot
    def shap_summary(self, model, df_type='train'):
        model.fit(self.train.drop(self.target, axis=1), self.train[self.target])
        if df_type=='train':
            explainer = shap.Explainer(model.predict, self.train.drop(self.target, axis=1))
            shap_values = explainer(self.train.drop(self.target, axis=1))
            return shap.plots.bar(shap_values, show=False)

        elif df_type=='test':
            explainer = shap.Explainer(model.predict, self.test.drop(self.target, axis=1))
            shap_values = explainer(self.test.drop(self.target, axis=1))
            return shap.plots.bar(shap_values, show=False)

        else:
            explainer = shap.Explainer(model.predict, self.df.drop(self.target, axis=1))
            shap_values = explainer(self.df.drop(self.target, axis=1))
            return shap.plots.bar(shap_values, show=False)


    def shap_scatter_1d(self, model, variable, df_type='train'):
        model.fit(self.train.drop(self.target, axis=1), self.train[self.target])
        if df_type=='train':
            explainer = shap.Explainer(model.predict, self.train.drop(self.target, axis=1))
            shap_values = explainer(self.train.drop(self.target, axis=1))
            ax = shap.plots.scatter(shap_values[:, variable], hist=False, show=False)
            return ax
        elif df_type=='test':
            explainer = shap.Explainer(model.predict, self.test.drop(self.target, axis=1))
            shap_values = explainer(self.test.drop(self.target, axis=1))
            ax = shap.plots.scatter(shap_values[:, variable], hist=False, show=False)
            return ax
        else:
            explainer = shap.Explainer(model.predict, self.df.drop(self.target, axis=1))
            shap_values = explainer(self.df.drop(self.target, axis=1))
            # ax = shap.plots.scatter(shap_values[:, variable], hist=False, show=False, x_jitter=0)

            fig, ax = plt.subplots()
            ax.scatter(shap_values[:, variable].data,
                       shap_values[:, variable].values,
                       s=10,
                       label='SHAP values')
            ax.set_xlabel(variable)
            ax.set_ylabel(f'SHAP values for {variable}')

            # LOWESS curve fit
            result = sm.nonparametric.lowess(shap_values[:, variable].values,
                                             shap_values[:, variable].data,
                                             frac=0.2, it=3, delta=0.0)

            # 找零点
            lowess_x, lowess_y = result[:, 0], result[:, 1]
            zero_points = find_zeros_by_interpolation(lowess_x, lowess_y)

            for x_zero in zero_points:
                # ax.axvline(x=x_zero, color='red', linestyle='--', alpha=0.5)
                ax.annotate(f'x={x_zero:.3f}',
                            xy=(x_zero, 0),
                            xytext=(5, 20),
                            textcoords='offset points',
                            ha='center', fontsize=9,
                            arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

            ax.plot(result[:, 0], result[:, 1],
                    color='red', linewidth=2,
                    label='LOWESS Fit', zorder=10)
            ax.axhline(y=0, color='black', linewidth=1, linestyle='--', alpha=0.5, label='SHAP=0')
            return ax

    def shap_scatter_2d(self, model, variable, interact_term, df_type='all'):
        model.fit(self.train.drop(self.target, axis=1), self.train[self.target])
        if df_type=='train':
            explainer = shap.Explainer(model.predict, self.train.drop(self.target, axis=1))
            shap_values = explainer(self.train.drop(self.target, axis=1))
            ax = shap.plots.scatter(shap_values[:, variable], color=shap_values[:, interact_term], alpha=0.9, hist=False, show=False, x_jitter=0.1)
            return ax

        elif df_type=='test':
            explainer = shap.Explainer(model.predict, self.test.drop(self.target, axis=1))
            shap_values = explainer(self.test.drop(self.target, axis=1))
            ax = shap.plots.scatter(shap_values[:, variable], color=shap_values[:, interact_term], alpha=0.9, hist=False, show=False, x_jitter=0.1)
            return ax

        else:
            explainer = shap.Explainer(model, self.df.drop(self.target, axis=1))
            shap_values = explainer(self.df.drop(self.target, axis=1))
            ax = shap.plots.scatter(shap_values[:, variable], color=shap_values[:, interact_term], alpha=0.9, hist=False, show=False, x_jitter=0.1)
            return ax

def main():
    df = pd.read_csv('data.csv')
    target = 'Compressive strength'
    fi = Importance(df, target)
    params = {'n_estimators': 85, 'max_features': 0.2728120566947818}
    # fi.pfi(model=ExtraTreesRegressor(**params), df_type='all')
    # fi.shap_summary(model=ExtraTreesRegressor(**params), df_type='all')
    # fi.pdp(model=ExtraTreesRegressor(**params), features=[('RBA replacement ratio', 'Basalt fiber')], df_type='all')

    # fi.shap_beeswarm(model=ExtraTreesRegressor(**params), df_type='train')
    # fi.shap_scatter_1d(model=ExtraTreesRegressor(**params), variable='RBA replacement ratio', df_type='all')
    fi.shap_scatter_2d(model=ExtraTreesRegressor(**params), variable='RBA replacement ratio', interact_term='Basalt fiber', df_type='all')
    plt.tight_layout()
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
