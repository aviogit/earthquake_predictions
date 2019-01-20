import numpy as np
import pandas as pd
import csv
from typing import List, Tuple

from sklearn.linear_model import LinearRegression


def _get_trend(data: pd.core.frame.DataFrame, abs=False):
    ids = np.array(range(len(data)))
    array = np.abs(data.values) if abs else data.values

    linear_regression = LinearRegression()
    linear_regression.fit(ids.reshape(-1, 1), array)
    return linear_regression.coef_[0]


def _get_features(data: pd.core.frame.DataFrame):
    mean = data.mean()
    std = data.std()
    minimum = data.min()
    maximum = data.max()

    abs = np.abs(data)
    abs_mean = abs.mean()
    abs_std = abs.std()
    abs_min = abs.min()
    abs_max = abs.max()

    q95 = np.quantile(data, 0.95)
    q99 = np.quantile(data, 0.99)
    q05 = np.quantile(data, 0.05)
    q01 = np.quantile(data, 0.01)

    trend = _get_trend(data)
    trend_abs = _get_trend(data, True)

    return np.array([mean, std, minimum, maximum,
                     abs_mean, abs_std, abs_min, abs_max,
                     q95, q99, q05, q01,
                     trend, trend_abs])


def get_stat_summaries(data: pd.core.frame.DataFrame, aggregate_length: int = 5000, include_y: bool = True):
    size = len(data)
    samples = np.zeros((0, 15 if include_y else 14))

    for i in range(0, size, aggregate_length):
        step_data = data[i:i + aggregate_length]
        new_row = _get_features(step_data.iloc[:, 0])
        if include_y:
            new_row = np.append(new_row, step_data.iloc[-1, 1])

        samples = np.vstack([samples, new_row])

    columns = ['mean', 'std', 'min', 'max',
               'abs_mean', 'abs_std', 'abs_min', 'abs_max',
               'q95', 'q99', 'q05', 'q01',
               'trend', 'trend_abs']

    if include_y:
        columns.append('time_to_failure')

    return pd.DataFrame(dtype=np.float64, columns=columns, data=samples)


def split_sequence(file_path: str):
    counter = 0
    last_time_to_failure = 1e100
    tuples = []

    with open(file_path) as data:
        data.readline()  # skip headers
        for line in data:
            signal, time_to_failure = line.split(',')
            signal, time_to_failure = int(signal), float(time_to_failure)

            if time_to_failure > last_time_to_failure:
                export_sequence(counter, tuples)
                tuples = []
                counter += 1

            tuples.append((signal, time_to_failure))
            last_time_to_failure = time_to_failure


def export_sequence(id: int, data: List[Tuple[int, float]]):
    with open('../data/train{0}.csv'.format(id), mode='w') as target_file:
        writer = csv.writer(target_file, delimiter=',')

        for i in range(len(data)):
            writer.writerow([data[i][0], data[i][1]])
