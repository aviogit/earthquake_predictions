import os
import sys
import psutil

import numpy as np
import scipy
import pandas as pd
import csv
from typing import List, Tuple

from sklearn.linear_model import LinearRegression
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve

from scipy.signal import stft
from scipy.fftpack import fft


from joblib import Parallel, delayed

np.seterr(divide='ignore', invalid='ignore')


def _get_trend(data: pd.core.frame.DataFrame, abs=False):
    ids = np.array(range(len(data)))
    array = np.abs(data.values) if abs else data.values

    linear_regression = LinearRegression()
    linear_regression.fit(ids.reshape(-1, 1), array)
    return linear_regression.coef_[0]


def _append_features(index: int, stat_summary: pd.core.frame.DataFrame, step_data: pd.core.frame.DataFrame, windows_list, do_fft, do_stft, step_size_large=50000, step_size_small=1000, debug=False):
    stat_summary.loc[index, 'mean'] = step_data.mean()
    stat_summary.loc[index, 'std'] = step_data.std()
    stat_summary.loc[index, 'min'] = step_data.min()
    stat_summary.loc[index, 'max'] = step_data.max()

    if do_fft:
        f        = fft(step_data)
        f_amp    = np.sqrt(np.real(f)**2 + np.imag(f)**2)
        fft_data = pd.Series(f_amp)

        '''
        print(fft_data)
        print(step_data)
        print(f_amp.shape)
        print(fft_data.shape)
        print(fft_data.mean())
        print(step_data.shape)
        print(step_data.mean())
        '''

#        print(index, stat_summary.index)
#        print(index, stat_summary.columns)
#        #print(stat_summary.describe())
#        print(index, stat_summary)

        stat_summary.loc[index, 'fft_mean']             = fft_data.mean()
        stat_summary.loc[index, 'fft_std']              = fft_data.std()
        stat_summary.loc[index, 'fft_min']              = fft_data.min()
        stat_summary.loc[index, 'fft_max']              = fft_data.max()

        stat_summary.loc[index, 'fft_q95']              = np.quantile(fft_data, 0.95)
        stat_summary.loc[index, 'fft_q99']              = np.quantile(fft_data, 0.99)
        stat_summary.loc[index, 'fft_q05']              = np.quantile(fft_data, 0.05)
        stat_summary.loc[index, 'fft_q01']              = np.quantile(fft_data, 0.01)

        stat_summary.loc[index, 'fft_std_first5k']      = fft_data[:step_size_large].mean()
        stat_summary.loc[index, 'fft_mean_first5k']     = fft_data[:step_size_large].std()
        stat_summary.loc[index, 'fft_min_first5k']      = fft_data[:step_size_large].min()
        stat_summary.loc[index, 'fft_max_first5k']      = fft_data[:step_size_large].max()

        stat_summary.loc[index, 'fft_std_last5k']       = fft_data[-step_size_large:].mean()
        stat_summary.loc[index, 'fft_mean_last5k']      = fft_data[-step_size_large:].std()
        stat_summary.loc[index, 'fft_min_last5k']       = fft_data[-step_size_large:].min()
        stat_summary.loc[index, 'fft_max_last5k']       = fft_data[-step_size_large:].max()

        stat_summary.loc[index, 'fft_std_first1k']      = fft_data[:step_size_small].mean()
        stat_summary.loc[index, 'fft_mean_first1k']     = fft_data[:step_size_small].std()
        stat_summary.loc[index, 'fft_min_first1k']      = fft_data[:step_size_small].min()
        stat_summary.loc[index, 'fft_max_first1k']      = fft_data[:step_size_small].max()

        stat_summary.loc[index, 'fft_std_last1k']       = fft_data[-step_size_small:].mean()
        stat_summary.loc[index, 'fft_mean_last1k']      = fft_data[-step_size_small:].std()
        stat_summary.loc[index, 'fft_min_last1k']       = fft_data[-step_size_small:].min()
        stat_summary.loc[index, 'fft_max_last1k']       = fft_data[-step_size_small:].max()

        stat_summary.loc[index, 'fft_trend']            = _get_trend(fft_data)
        stat_summary.loc[index, 'fft_trend_abs']        = _get_trend(fft_data, True)

        stat_summary.loc[index, 'fft_count_big']        = len(fft_data[np.abs(fft_data) > 500])
        stat_summary.loc[index, 'fft_hilbert_mean']     = np.abs(hilbert(fft_data)).mean()

        hann_150 = hann(150)
        stat_summary.loc[index, 'fft_hann_window_mean'] = (convolve(fft_data, hann_150, mode='same') / sum(hann_150)).mean()


    if do_stft:
        #f        = fft(step_data)
        #f_amp    = np.sqrt(np.real(f)**2 + np.imag(f)**2)
        #fft_data = pd.Series(f_amp)
        stft_data, stft_data_t, stft_data_Zxx = stft(step_data, nperseg=step_data.shape[0])
        stft_data = pd.Series(stft_data)

        stat_summary.loc[index, 'stft_mean']             = stft_data.mean()
        stat_summary.loc[index, 'stft_std']              = stft_data.std()
        stat_summary.loc[index, 'stft_min']              = stft_data.min()
        stat_summary.loc[index, 'stft_max']              = stft_data.max()

        stat_summary.loc[index, 'stft_q95']              = np.quantile(stft_data, 0.95)
        stat_summary.loc[index, 'stft_q99']              = np.quantile(stft_data, 0.99)
        stat_summary.loc[index, 'stft_q05']              = np.quantile(stft_data, 0.05)
        stat_summary.loc[index, 'stft_q01']              = np.quantile(stft_data, 0.01)

        stat_summary.loc[index, 'stft_std_first5k']      = stft_data[:step_size_large].mean()
        stat_summary.loc[index, 'stft_mean_first5k']     = stft_data[:step_size_large].std()
        stat_summary.loc[index, 'stft_min_first5k']      = stft_data[:step_size_large].min()
        stat_summary.loc[index, 'stft_max_first5k']      = stft_data[:step_size_large].max()

        stat_summary.loc[index, 'stft_std_last5k']       = stft_data[-step_size_large:].mean()
        stat_summary.loc[index, 'stft_mean_last5k']      = stft_data[-step_size_large:].std()
        stat_summary.loc[index, 'stft_min_last5k']       = stft_data[-step_size_large:].min()
        stat_summary.loc[index, 'stft_max_last5k']       = stft_data[-step_size_large:].max()

        stat_summary.loc[index, 'stft_std_first1k']      = stft_data[:step_size_small].mean()
        stat_summary.loc[index, 'stft_mean_first1k']     = stft_data[:step_size_small].std()
        stat_summary.loc[index, 'stft_min_first1k']      = stft_data[:step_size_small].min()
        stat_summary.loc[index, 'stft_max_first1k']      = stft_data[:step_size_small].max()

        stat_summary.loc[index, 'stft_std_last1k']       = stft_data[-step_size_small:].mean()
        stat_summary.loc[index, 'stft_mean_last1k']      = stft_data[-step_size_small:].std()
        stat_summary.loc[index, 'stft_min_last1k']       = stft_data[-step_size_small:].min()
        stat_summary.loc[index, 'stft_max_last1k']       = stft_data[-step_size_small:].max()

        stat_summary.loc[index, 'stft_trend']            = _get_trend(stft_data)
        stat_summary.loc[index, 'stft_trend_abs']        = _get_trend(stft_data, True)

        stat_summary.loc[index, 'stft_count_big']        = len(stft_data[np.abs(stft_data) > 500])
        stat_summary.loc[index, 'stft_hilbert_mean']     = np.abs(hilbert(stft_data)).mean()

        hann_150 = hann(150)
        stat_summary.loc[index, 'stft_hann_window_mean'] = (convolve(stft_data, hann_150, mode='same') / sum(hann_150)).mean()


    absolutes = np.abs(step_data)

    stat_summary.loc[index, 'abs_mean'] = absolutes.mean()
    stat_summary.loc[index, 'abs_std'] = absolutes.std()
    stat_summary.loc[index, 'abs_min'] = absolutes.min()
    stat_summary.loc[index, 'abs_max'] = absolutes.max()

    stat_summary.loc[index, 'q95'] = np.quantile(step_data, 0.95)
    stat_summary.loc[index, 'q99'] = np.quantile(step_data, 0.99)
    stat_summary.loc[index, 'q05'] = np.quantile(step_data, 0.05)
    stat_summary.loc[index, 'q01'] = np.quantile(step_data, 0.01)

    stat_summary.loc[index, 'std_first5k'] = step_data[:step_size_large].mean()
    stat_summary.loc[index, 'mean_first5k'] = step_data[:step_size_large].std()
    stat_summary.loc[index, 'min_first5k'] = step_data[:step_size_large].min()
    stat_summary.loc[index, 'max_first5k'] = step_data[:step_size_large].max()

    stat_summary.loc[index, 'std_last5k'] = step_data[-step_size_large:].mean()
    stat_summary.loc[index, 'mean_last5k'] = step_data[-step_size_large:].std()
    stat_summary.loc[index, 'min_last5k'] = step_data[-step_size_large:].min()
    stat_summary.loc[index, 'max_last5k'] = step_data[-step_size_large:].max()

    stat_summary.loc[index, 'std_first1k'] = step_data[:step_size_small].mean()
    stat_summary.loc[index, 'mean_first1k'] = step_data[:step_size_small].std()
    stat_summary.loc[index, 'min_first1k'] = step_data[:step_size_small].min()
    stat_summary.loc[index, 'max_first1k'] = step_data[:step_size_small].max()

    stat_summary.loc[index, 'std_last1k'] = step_data[-step_size_small:].mean()
    stat_summary.loc[index, 'mean_last1k'] = step_data[-step_size_small:].std()
    stat_summary.loc[index, 'min_last1k'] = step_data[-step_size_small:].min()
    stat_summary.loc[index, 'max_last1k'] = step_data[-step_size_small:].max()

    stat_summary.loc[index, 'trend'] = _get_trend(step_data)
    stat_summary.loc[index, 'trend_abs'] = _get_trend(step_data, True)

    stat_summary.loc[index, 'count_big'] = len(step_data[np.abs(step_data) > 500])
    stat_summary.loc[index, 'hilbert_mean'] = np.abs(hilbert(step_data)).mean()

    hann_150 = hann(150)
    stat_summary.loc[index, 'hann_window_mean'] = (convolve(step_data, hann_150, mode='same') / sum(hann_150)).mean()




    # Found here: https://www.kaggle.com/amignan/baseline-rf-model-reproducing-the-2017-paper

    records = len(step_data.index)
    stat_summary.loc[index, 'meanA'] = step_data[0:round(records/2)].mean()
    stat_summary.loc[index, 'meanB'] = step_data[round(records/2)+1:records].mean()
    stat_summary.loc[index, 'varA'] = step_data[0:round(records/2)].var()
    stat_summary.loc[index, 'varB'] = step_data[round(records/2)+1:records].var()
    stat_summary.loc[index, 'skewA'] = scipy.stats.skew(step_data[0:round(records/2)])
    stat_summary.loc[index, 'skewB'] = scipy.stats.skew(step_data[round(records/2)+1:records])
    stat_summary.loc[index, 'kurtA'] = scipy.stats.kurtosis(step_data[0:round(records/2)])
    stat_summary.loc[index, 'kurtB'] = scipy.stats.kurtosis(step_data[round(records/2)+1:records])
    stat_summary.loc[index, 'varAnorm'] = stat_summary.loc[index, 'varA']/(stat_summary.loc[index, 'varA']+stat_summary.loc[index, 'varB'])
    stat_summary.loc[index, 'varBnorm'] = stat_summary.loc[index, 'varB']/(stat_summary.loc[index, 'varA']+stat_summary.loc[index, 'varB'])
    stat_summary.loc[index, 'skewAnorm'] = stat_summary.loc[index, 'skewA']/(stat_summary.loc[index, 'skewA']+stat_summary.loc[index, 'skewB'])
    stat_summary.loc[index, 'skewBnorm'] = stat_summary.loc[index, 'skewB']/(stat_summary.loc[index, 'skewA']+stat_summary.loc[index, 'skewB'])
    stat_summary.loc[index, 'kurtAnorm'] = stat_summary.loc[index, 'kurtA']/(stat_summary.loc[index, 'kurtA']+stat_summary.loc[index, 'kurtB'])
    stat_summary.loc[index, 'kurtBnorm'] = stat_summary.loc[index, 'kurtB']/(stat_summary.loc[index, 'kurtA']+stat_summary.loc[index, 'kurtB'])

    stat_summary.loc[index, 'q01A'] = np.quantile(step_data[0:round(records/2)], 0.01)
    stat_summary.loc[index, 'q02A'] = np.quantile(step_data[0:round(records/2)], 0.02)
    stat_summary.loc[index, 'q03A'] = np.quantile(step_data[0:round(records/2)], 0.03)
    stat_summary.loc[index, 'q04A'] = np.quantile(step_data[0:round(records/2)], 0.04)
    stat_summary.loc[index, 'q05A'] = np.quantile(step_data[0:round(records/2)], 0.05)
    stat_summary.loc[index, 'q06A'] = np.quantile(step_data[0:round(records/2)], 0.06)
    stat_summary.loc[index, 'q07A'] = np.quantile(step_data[0:round(records/2)], 0.07)
    stat_summary.loc[index, 'q08A'] = np.quantile(step_data[0:round(records/2)], 0.08)
    stat_summary.loc[index, 'q09A'] = np.quantile(step_data[0:round(records/2)], 0.09)
    stat_summary.loc[index, 'q91A'] = np.quantile(step_data[0:round(records/2)], 0.91)
    stat_summary.loc[index, 'q92A'] = np.quantile(step_data[0:round(records/2)], 0.92)
    stat_summary.loc[index, 'q93A'] = np.quantile(step_data[0:round(records/2)], 0.93)
    stat_summary.loc[index, 'q94A'] = np.quantile(step_data[0:round(records/2)], 0.94)
    stat_summary.loc[index, 'q95A'] = np.quantile(step_data[0:round(records/2)], 0.95)
    stat_summary.loc[index, 'q96A'] = np.quantile(step_data[0:round(records/2)], 0.96)
    stat_summary.loc[index, 'q97A'] = np.quantile(step_data[0:round(records/2)], 0.97)
    stat_summary.loc[index, 'q98A'] = np.quantile(step_data[0:round(records/2)], 0.98)
    stat_summary.loc[index, 'q99A'] = np.quantile(step_data[0:round(records/2)], 0.99)
    stat_summary.loc[index, 'q01B'] = np.quantile(step_data[round(records/2)+1:records], 0.01)
    stat_summary.loc[index, 'q02B'] = np.quantile(step_data[round(records/2)+1:records], 0.02)
    stat_summary.loc[index, 'q03B'] = np.quantile(step_data[round(records/2)+1:records], 0.03)
    stat_summary.loc[index, 'q04B'] = np.quantile(step_data[round(records/2)+1:records], 0.04)
    stat_summary.loc[index, 'q05B'] = np.quantile(step_data[round(records/2)+1:records], 0.05)
    stat_summary.loc[index, 'q06B'] = np.quantile(step_data[round(records/2)+1:records], 0.06)
    stat_summary.loc[index, 'q07B'] = np.quantile(step_data[round(records/2)+1:records], 0.07)
    stat_summary.loc[index, 'q08B'] = np.quantile(step_data[round(records/2)+1:records], 0.08)
    stat_summary.loc[index, 'q09B'] = np.quantile(step_data[round(records/2)+1:records], 0.09)
    stat_summary.loc[index, 'q91B'] = np.quantile(step_data[round(records/2)+1:records], 0.91)
    stat_summary.loc[index, 'q92B'] = np.quantile(step_data[round(records/2)+1:records], 0.92)
    stat_summary.loc[index, 'q93B'] = np.quantile(step_data[round(records/2)+1:records], 0.93)
    stat_summary.loc[index, 'q94B'] = np.quantile(step_data[round(records/2)+1:records], 0.94)
    stat_summary.loc[index, 'q95B'] = np.quantile(step_data[round(records/2)+1:records], 0.95)
    stat_summary.loc[index, 'q96B'] = np.quantile(step_data[round(records/2)+1:records], 0.96)
    stat_summary.loc[index, 'q97B'] = np.quantile(step_data[round(records/2)+1:records], 0.97)
    stat_summary.loc[index, 'q98B'] = np.quantile(step_data[round(records/2)+1:records], 0.98)
    stat_summary.loc[index, 'q99B'] = np.quantile(step_data[round(records/2)+1:records], 0.99)

    f0pos = (1e-9, 5e-9, 1e-8, 5e-8, 1e-7)
    f0neg = (-1e-9, -5e-9, -1e-8, -5e-8, -1e-7)

    stat_summary.loc[index, 'f00pA'] = sum(step_data[0:round(records/2)] >= f0pos[0])/75000
    stat_summary.loc[index, 'f01pA'] = sum(step_data[0:round(records/2)] >= f0pos[1])/75000
    stat_summary.loc[index, 'f02pA'] = sum(step_data[0:round(records/2)] >= f0pos[2])/75000
    stat_summary.loc[index, 'f03pA'] = sum(step_data[0:round(records/2)] >= f0pos[3])/75000
    stat_summary.loc[index, 'f04pA'] = sum(step_data[0:round(records/2)] >= f0pos[4])/75000
    stat_summary.loc[index, 'f00nA'] = sum(step_data[0:round(records/2)] <= f0neg[0])/75000
    stat_summary.loc[index, 'f01nA'] = sum(step_data[0:round(records/2)] <= f0neg[1])/75000
    stat_summary.loc[index, 'f02nA'] = sum(step_data[0:round(records/2)] <= f0neg[2])/75000
    stat_summary.loc[index, 'f03nA'] = sum(step_data[0:round(records/2)] <= f0neg[3])/75000
    stat_summary.loc[index, 'f04nA'] = sum(step_data[0:round(records/2)] <= f0neg[4])/75000
    stat_summary.loc[index, 'f00pB'] = sum(step_data[round(records/2)+1:records] >= f0pos[0])/74999
    stat_summary.loc[index, 'f01pB'] = sum(step_data[round(records/2)+1:records] >= f0pos[1])/74999
    stat_summary.loc[index, 'f02pB'] = sum(step_data[round(records/2)+1:records] >= f0pos[2])/74999
    stat_summary.loc[index, 'f03pB'] = sum(step_data[round(records/2)+1:records] >= f0pos[3])/74999
    stat_summary.loc[index, 'f04pB'] = sum(step_data[round(records/2)+1:records] >= f0pos[4])/74999
    stat_summary.loc[index, 'f00nB'] = sum(step_data[round(records/2)+1:records] <= f0neg[0])/74999
    stat_summary.loc[index, 'f01nB'] = sum(step_data[round(records/2)+1:records] <= f0neg[1])/74999
    stat_summary.loc[index, 'f02nB'] = sum(step_data[round(records/2)+1:records] <= f0neg[2])/74999
    stat_summary.loc[index, 'f03nB'] = sum(step_data[round(records/2)+1:records] <= f0neg[3])/74999
    stat_summary.loc[index, 'f04nB'] = sum(step_data[round(records/2)+1:records] <= f0neg[4])/74999
    
    stat_summary.loc[index, 'minA'] = min(step_data[0:round(records/2)])
    stat_summary.loc[index, 'maxA'] = max(step_data[0:round(records/2)])
    stat_summary.loc[index, 'minB'] = min(step_data[round(records/2)+1:records])
    stat_summary.loc[index, 'maxB'] = max(step_data[round(records/2)+1:records])










    for windows in windows_list:
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
        if debug:
            print('Job['+str(int(index))+'] - roll_mean:', roll_mean, ' - len:', len(roll_mean))
            np.savetxt('/tmp/roll-means/roll-mean-idx-'+str(int(index))+'-win-'+str(int(windows))+'.npy', roll_mean, fmt='%.5e')

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


def _append_features_wrapper(data, aggregate_length, do_fft, do_stft, i, stat_summary, include_y, windows_list):
	process = psutil.Process(os.getpid())
	ram	= process.memory_info()[0] / float(2 ** 20)

	index = i/aggregate_length
	print('[' + str(i) + '] Running job with index:', str(int(index)), '/', len(data)/aggregate_length, ' - used RAM:', ram)

	step_data = data[i:i + aggregate_length]

	step_data_mean    = step_data.mean()
	step_data_mean[1] = 0

	print(f'Subtracting mean value {step_data_mean[0]} from the acoustic samples in the current chunk...')
	step_data = step_data - step_data_mean

	_append_features(index, stat_summary, step_data.iloc[:, 0], windows_list, do_fft, do_stft)

	if include_y:
		stat_summary.loc[index, 'time_to_failure'] = step_data.iloc[-1, 1]

	del step_data					# try to free some memory

def get_stat_summaries(data: pd.core.frame.DataFrame, aggregate_length: int = 150000, do_fft = True, do_stft = True, run_parallel = True, include_y: bool = True, debug=False):
    size = len(data)
    windows_list = [10, 100, 1000]






    if run_parallel:
         # These are fine
         cols = [ 'mean', 'std', 'min', 'max']

         if do_fft:
              cols.extend([ 'fft_mean', 'fft_std', 'fft_min', 'fft_max', 'fft_q95', 'fft_q99', 'fft_q05', 'fft_q01', 'fft_std_first5k', 'fft_mean_first5k', 'fft_min_first5k', 'fft_max_first5k', 'fft_std_last5k', 'fft_mean_last5k', 'fft_min_last5k', 'fft_max_last5k', 'fft_std_first1k', 'fft_mean_first1k', 'fft_min_first1k', 'fft_max_first1k', 'fft_std_last1k', 'fft_mean_last1k', 'fft_min_last1k', 'fft_max_last1k', 'fft_trend', 'fft_trend_abs', 'fft_count_big', 'fft_hilbert_mean', 'fft_hann_window_mean' ])

         if do_stft:
              cols.extend([ 'stft_mean', 'stft_std', 'stft_min', 'stft_max', 'stft_q95', 'stft_q99', 'stft_q05', 'stft_q01', 'stft_std_first5k', 'stft_mean_first5k', 'stft_min_first5k', 'stft_max_first5k', 'stft_std_last5k', 'stft_mean_last5k', 'stft_min_last5k', 'stft_max_last5k', 'stft_std_first1k', 'stft_mean_first1k', 'stft_min_first1k', 'stft_max_first1k', 'stft_std_last1k', 'stft_mean_last1k', 'stft_min_last1k', 'stft_max_last1k', 'stft_trend', 'stft_trend_abs', 'stft_count_big', 'stft_hilbert_mean', 'stft_hann_window_mean' ])

         cols.extend([ 'abs_mean', 'abs_std', 'abs_min', 'abs_max', 'q95', 'q99', 'q05', 'q01', 'std_first5k', 'mean_first5k', 'min_first5k', 'max_first5k', 'std_last5k', 'mean_last5k', 'min_last5k', 'max_last5k', 'std_first1k', 'mean_first1k', 'min_first1k', 'max_first1k', 'std_last1k', 'mean_last1k', 'min_last1k', 'max_last1k', 'trend', 'trend_abs', 'count_big', 'hilbert_mean', 'hann_window_mean'])

         # Found here: https://www.kaggle.com/amignan/baseline-rf-model-reproducing-the-2017-paper
         cols.extend(['meanA', 'varA', 'varAnorm', 'skewA', 'skewAnorm', 'kurtA', 'kurtAnorm', 'meanB', 'varB', 'varBnorm', 'skewB', 'skewBnorm', 'kurtB', 'kurtBnorm', 'q01A', 'q02A', 'q03A', 'q04A', 'q05A', 'q06A', 'q07A', 'q08A', 'q09A', 'q01B', 'q02B', 'q03B', 'q04B', 'q05B', 'q06B', 'q07B', 'q08B', 'q09B', 'q91A', 'q92A', 'q93A', 'q94A', 'q95A', 'q96A', 'q97A', 'q98A', 'q99A', 'q91B', 'q92B', 'q93B', 'q94B', 'q95B', 'q96B', 'q97B', 'q98B', 'q99B', 'f00pA', 'f01pA', 'f02pA', 'f03pA', 'f04pA', 'f00nA', 'f01nA', 'f02nA', 'f03nA', 'f04nA', 'f00pB', 'f01pB', 'f02pB', 'f03pB', 'f04pB', 'f00nB', 'f01nB', 'f02nB', 'f03nB', 'f04nB', 'minA', 'maxA', 'minB', 'maxB'])


         # These need to be concat with windows_str ([10, 100, 1000] - see above)
         cols_param = [ 'mean_roll_std', 'std_roll_std', 'min_roll_std', 'max_roll_std', 'q95_roll_std', 'q99_roll_std', 'q05_roll_std', 'q01_roll_std', 'change_abs_roll_std', 'change_rate_roll_std', 'mean_roll_mean', 'std_roll_mean', 'min_roll_mean', 'max_roll_mean', 'q95_roll_mean', 'q99_roll_mean', 'q05_roll_mean', 'q01_roll_mean', 'change_abs_roll_mean', 'change_rate_roll_mean']

         for i in windows_list:
              for j in cols_param:
                   cols.append(j+str(i))

         # I don't know how I've been so dumb to put time_to_failure in the middle of other columns :(
         cols.append('time_to_failure')

         #debug = True
         if debug:
              print(cols)

         # pre-alloc columns, so that Pandas doesn't throw keyerror running in parallel
         stat_summary = pd.DataFrame(np.nan, index=np.arange(0, size/aggregate_length), columns=cols, dtype=np.float64)
#         print(stat_summary.index)
#         print(stat_summary.columns)
#         #print(stat_summary.describe())
#         print(stat_summary)

         Parallel(n_jobs=8*4, prefer='threads')(
		delayed(_append_features_wrapper)(data, aggregate_length, do_fft, do_stft, i, stat_summary, include_y, windows_list)
		for i in range(0, size, aggregate_length))
    else:

         stat_summary = pd.DataFrame(index=np.arange(0, size/aggregate_length), dtype=np.float64)

         for i in range(0, size, aggregate_length):
              _append_features_wrapper(data, aggregate_length, do_fft, do_stft, i, stat_summary, include_y, windows_list)

    if debug:
         print('stat_summary len', len(stat_summary.index))
    print('stat_summary', stat_summary)

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
