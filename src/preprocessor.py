import numpy as np
import pandas as pd
import csv
from typing import List, Tuple

from sklearn.linear_model import LinearRegression
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve

np.seterr(divide='ignore', invalid='ignore')


def _get_trend(data: pd.core.frame.DataFrame, abs=False):
    ids = np.array(range(len(data)))
    array = np.abs(data.values) if abs else data.values

    linear_regression = LinearRegression()
    linear_regression.fit(ids.reshape(-1, 1), array)
    return linear_regression.coef_[0]


def _append_features(index: int, stat_summary: pd.core.frame.DataFrame, step_data: pd.core.frame.DataFrame):
    stat_summary.loc[index, 'mean'] = step_data.mean()
    stat_summary.loc[index, 'std'] = step_data.std()
    stat_summary.loc[index, 'min'] = step_data.min()
    stat_summary.loc[index, 'max'] = step_data.max()

    absolutes = np.abs(step_data)

    stat_summary.loc[index, 'abs_mean'] = absolutes.mean()
    stat_summary.loc[index, 'abs_std'] = absolutes.std()
    stat_summary.loc[index, 'abs_min'] = absolutes.min()
    stat_summary.loc[index, 'abs_max'] = absolutes.max()

    stat_summary.loc[index, 'q95'] = np.quantile(step_data, 0.95)
    stat_summary.loc[index, 'q99'] = np.quantile(step_data, 0.99)
    stat_summary.loc[index, 'q05'] = np.quantile(step_data, 0.05)
    stat_summary.loc[index, 'q01'] = np.quantile(step_data, 0.01)

    stat_summary.loc[index, 'std_first5k'] = step_data[:50000].mean()
    stat_summary.loc[index, 'mean_first5k'] = step_data[:50000].std()
    stat_summary.loc[index, 'min_first5k'] = step_data[:50000].min()
    stat_summary.loc[index, 'max_first5k'] = step_data[:50000].max()

    stat_summary.loc[index, 'std_last5k'] = step_data[-50000:].mean()
    stat_summary.loc[index, 'mean_last5k'] = step_data[-50000:].std()
    stat_summary.loc[index, 'min_last5k'] = step_data[-50000:].min()
    stat_summary.loc[index, 'max_last5k'] = step_data[-50000:].max()

    stat_summary.loc[index, 'std_first1k'] = step_data[:1000].mean()
    stat_summary.loc[index, 'mean_first1k'] = step_data[:1000].std()
    stat_summary.loc[index, 'min_first1k'] = step_data[:1000].min()
    stat_summary.loc[index, 'max_first1k'] = step_data[:1000].max()

    stat_summary.loc[index, 'std_last1k'] = step_data[-1000:].mean()
    stat_summary.loc[index, 'mean_last1k'] = step_data[-1000:].std()
    stat_summary.loc[index, 'min_last1k'] = step_data[-1000:].min()
    stat_summary.loc[index, 'max_last1k'] = step_data[-1000:].max()

    stat_summary.loc[index, 'trend'] = _get_trend(step_data)
    stat_summary.loc[index, 'trend_abs'] = _get_trend(step_data, True)

    stat_summary.loc[index, 'count_big'] = len(step_data[np.abs(step_data) > 500])
    stat_summary.loc[index, 'hilbert_mean'] = np.abs(hilbert(step_data)).mean()

    hann_150 = hann(150)
    stat_summary.loc[index, 'hann_window_mean'] = (convolve(step_data, hann_150, mode='same') / sum(hann_150)).mean()

    for windows in [10, 100, 1000]:
        roll_std = step_data.rolling(windows).std().dropna().values
        windows_str = str(windows)

        stat_summary.loc[index, 'mean_roll_std' + windows_str] = roll_std.mean()
        stat_summary.loc[index, 'std_roll_std' + windows_str] = roll_std.std()
        stat_summary.loc[index, 'min_roll_std' + windows_str] = roll_std.min()
        stat_summary.loc[index, 'max_roll_std' + windows_str] = roll_std.max()

        stat_summary.loc[index, 'q95_roll_std' + windows_str] = np.quantile(roll_std, 0.95)
        stat_summary.loc[index, 'q99_roll_std' + windows_str] = np.quantile(roll_std, 0.99)
        stat_summary.loc[index, 'q05_roll_std' + windows_str] = np.quantile(roll_std, 0.05)
        stat_summary.loc[index, 'q01_roll_std' + windows_str] = np.quantile(roll_std, 0.01)

        stat_summary.loc[index, 'change_abs_roll_std' + windows_str] \
            = np.mean(np.nonzero((np.diff(roll_std) / roll_std[:-1]))[0])

        stat_summary.loc[index, 'change_rate_roll_std' + windows_str] = np.abs(roll_std).max()

        roll_mean = step_data.rolling(windows).mean().dropna().values

        stat_summary.loc[index, 'mean_roll_mean' + windows_str] = roll_mean.mean()
        stat_summary.loc[index, 'std_roll_mean' + windows_str] = roll_mean.std()
        stat_summary.loc[index, 'min_roll_mean' + windows_str] = roll_mean.min()
        stat_summary.loc[index, 'max_roll_mean' + windows_str] = roll_mean.max()

        stat_summary.loc[index, 'q95_roll_mean' + windows_str] = np.quantile(roll_mean, 0.95)
        stat_summary.loc[index, 'q99_roll_mean' + windows_str] = np.quantile(roll_mean, 0.99)
        stat_summary.loc[index, 'q05_roll_mean' + windows_str] = np.quantile(roll_mean, 0.05)
        stat_summary.loc[index, 'q01_roll_mean' + windows_str] = np.quantile(roll_mean, 0.01)

        stat_summary.loc[index, 'change_abs_roll_mean' + windows_str] \
            = np.mean(np.nonzero((np.diff(roll_mean) / roll_mean[:-1]))[0])

        stat_summary.loc[index, 'change_rate_roll_mean' + windows_str] = np.abs(roll_mean).max()


def get_stat_summaries(data: pd.core.frame.DataFrame, aggregate_length: int = 150000, include_y: bool = True):
    size = len(data)
    stat_summary = pd.DataFrame(dtype=np.float64)

    index = 0
    for i in range(0, size, aggregate_length):
        step_data = data[i:i + aggregate_length]
        _append_features(index, stat_summary, step_data.iloc[:, 0])

        if include_y:
            stat_summary.loc[index, 'time_to_failure'] = step_data.iloc[-1, 1]
        index += 1

    return stat_summary


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
                _export_sequence(counter, tuples)
                tuples = []
                counter += 1

            tuples.append((signal, time_to_failure))
            last_time_to_failure = time_to_failure


def _export_sequence(id: int, data: List[Tuple[int, float]]):
    with open('../data/train{0}.csv'.format(id), mode='w') as target_file:
        writer = csv.writer(target_file, delimiter=',')

        for i in range(len(data)):
            writer.writerow([data[i][0], data[i][1]])
