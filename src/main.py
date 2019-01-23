import numpy as np
import pandas as pd

from visualizer import save_plot_data
from preprocessor import get_stat_summaries
from feature_extractor import extract
from models.rnn import Rnn


def predict(model):
    submission = pd.read_csv(
        '../data/sample_submission.csv',
        index_col='seg_id',
        dtype={"time_to_failure": np.float32})

    for i, seg_id in enumerate(submission.index):
        # print('Predicting time to failure for {0}'.format(seg_id))
        seg = pd.read_csv('../data/test/' + seg_id + '.csv')
        summary = get_stat_summaries(seg, 150000, False)
        submission.time_to_failure[i] = model.predict(summary.values)

    submission.head()
    submission.to_csv('submission.csv')


def main():
    # training_set = pd.read_csv('data/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
    # visualizer.plot_data(training_set)
    # preprocessor.split_sequence('data/train.csv')
    training_set = pd.read_csv('../data/train.csv', dtype={'acoustic_data': np.float32, 'time_to_failure': np.float64})
    save_plot_data(training_set)

    summary = get_stat_summaries(training_set, 150000)
    summary.to_csv('../data/stat_summary.csv')

    training_set = summary.values
    feature_count = training_set.shape[-1] - 1

    # extract(summary.iloc[:, :-1], summary.iloc[:, -1])

    model = Rnn(feature_count)
    model.fit(training_set, batch_size=32, epochs=200)

    predict(model)


if __name__ == '__main__':
    main()
