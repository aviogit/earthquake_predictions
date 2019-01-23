import pandas as pd
import lightgbm as lgb

from sklearn.preprocessing import MinMaxScaler


def extract(x: pd.core.frame.DataFrame, y: pd.core.frame.DataFrame):
    params = {'num_leaves': 54,
              'min_data_in_leaf': 79,
              'objective': 'huber',
              'max_depth': -1,
              'learning_rate': 0.01,
              "boosting": "gbdt",
              # "feature_fraction": 0.8354507676881442,
              "bagging_freq": 3,
              "bagging_fraction": 0.8126672064208567,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 1.1302650970728192,
              'reg_lambda': 0.3603427518866501
              }

    scaler = MinMaxScaler(feature_range=(0, x.shape[1]))
    x_train, y_train = scaler.fit_transform(x.values), y.values

    lgb_train = lgb.DataSet(x_train, y_train)

    gbm = lgb.train(params, lgb_train)