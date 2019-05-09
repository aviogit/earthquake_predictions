#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import sys
import gzip
import io

from visualizer import save_summary_plot
from preprocessor import get_stat_summaries
from feature_extractor import extract
from models.rnn import Rnn

base_dir = '/tmp/LANL-Earthquake-Prediction-train-csv-gzipped'

def predict(model):
    submission = pd.read_csv(
        base_dir + '/sample_submission.csv',
        index_col='seg_id',
        dtype={"time_to_failure": np.float32})

    for i, seg_id in enumerate(submission.index):
        seg = pd.read_csv(base_dir + '/test/' + seg_id + '.csv')
        summary = get_stat_summaries(seg, 150000, run_parallel=False, include_y=False)
        submission.time_to_failure[i] = model.predict(summary.values)
        print('Prediction for submission no.:', i, ' - id: ', seg_id, ' - time to failure:', submission.time_to_failure[i])

    submission.head()
    submission.to_csv('submission.csv')


def main(argv):
    # training_set = pd.read_csv('data/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
    # visualizer.plot_data(training_set)
    # preprocessor.split_sequence('data/train.csv')

    # First, gzip the huge training set file, so to make it more manageable
    # gzip --to-stdout -1 train.csv > ~/LANL-Earthquake-Prediction-train-csv-splitted-gzip-1/train.csv.gz

    if len(argv) > 1:
        print('Loading features from file:', argv[1])
        summary = pd.read_csv(argv[1])

        summary['temp'] = summary['time_to_failure']
        summary.drop(columns=['time_to_failure'], inplace=True)
        summary['time_to_failure'] = summary['temp']
        summary.drop(columns=['temp'], inplace=True)
        summary.drop(columns=['Unnamed: 0'], inplace=True)

        print(summary)
        summary.to_csv(base_dir + '/stat_summary.csv')
    else:
        fname = base_dir + '/train.csv.gz'
        #fname = base_dir + '/LANL-Earthquake-Prediction-series-no-000.csv.gz'
        print('Opening and reading file:', fname)
        gzipped_file = gzip.open(fname, 'r')
        file_content = gzipped_file.read()

        print('Finished reading file, filling the DataFrame...')
        training_set = pd.read_csv(io.BytesIO(file_content), dtype={'acoustic_data': np.float32, 'time_to_failure': np.float64})
        #save_summary_plot(training_set)

        print('Extracting features...')
        summary = get_stat_summaries(training_set, 150000, run_parallel=True)
        summary.to_csv(base_dir + '/stat_summary.csv')
        print('Features have been saved to:', base_dir + '/stat_summary.csv')

    training_set = summary.values
    feature_count = training_set.shape[-1] - 1
    print(feature_count)
    print(training_set)
    #sys.exit(0)

    # extract(summary.iloc[:, :-1], summary.iloc[:, -1])

    print(20*'*', 'Start of training', 20*'*')
    model = Rnn(feature_count)
    model.fit(training_set, batch_size=32, epochs=500)
    print(20*'*', 'End of training', 20*'*')

    print(20*'*', 'Start of prediction ', 20*'*')
    predict(model)
    print(20*'*', 'End of prediction ', 20*'*')


if __name__ == '__main__':
    main(sys.argv)
