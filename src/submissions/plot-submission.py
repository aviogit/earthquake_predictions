#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import sys
import gzip
import io

def abs_error_df(submissions, sub_idx_good, new_sub_idx):
	abs_sub = (submissions[sub_idx_good] - submissions[new_sub_idx]).abs()
	ax = plt.gca()
	abs_sub.reset_index().plot(kind='line', y='time_to_failure', use_index=True, ax=ax, sharex=True)

def main(argv):
	if len(argv) <= 1:
		print('Error. Please provide at least one submission.csv file')
		sys.exit(0)

	ax = plt.gca()
	ax.set_xlabel("Test Sample")
	ax.set_ylabel("Seconds")

	idx = 0
	submissions = [None] * (len(argv)-1)
	legends = []

	#fig, axes = plt.subplots(nrows=1, ncols=3)
	for tok in argv:
		if tok == argv[0]:
			continue
		print("Reading argv:", tok)
		submissions[idx] = pd.read_csv(
			tok,
			index_col='seg_id',
			dtype={"time_to_failure": np.float32})
	
	
		# gca stands for 'get current axis'
		ax = plt.gca()

		legends.append(str(idx) + '-' + tok + '-TTF')
		
		submissions[idx].reset_index().plot(kind='line', y='time_to_failure', use_index=True, ax=ax, sharex=True)
		#submissions.plot(kind='line',x='name',y='num_pets', color='red', ax=ax)
		idx += 1

	if (len(argv)) > 2:
		abs_error_df(submissions, 0, 1)		# 1.564 vs. 1.578
		legends.append('abs-err-0-1')
	if (len(argv)) > 3:
		abs_error_df(submissions, 0, -1)
		abs_error_df(submissions, 1, -1)
		legends.extend(['abs-err-0-'+str(len(argv)-2), 'abs-err-1-'+str(len(argv)-2)])

	ax.legend(legends);
	plt.grid(True)
	plt.show()
	sys.exit(0)

'''
    for i, seg_id in enumerate(submission.index):
        seg = pd.read_csv(base_dir + '/test/' + seg_id + '.csv')
        summary = get_stat_summaries(seg, 4096, run_parallel=False, include_y=False)
        submission.time_to_failure[i] = model.predict(summary.values)
        print('Prediction for submission no.:', i, ' - id: ', seg_id, ' - time to failure:', submission.time_to_failure[i])

    submission.head()
    submission.to_csv('submission.csv')
'''

'''
def main(argv):

    if len(argv) > 1:
        print('Loading features from file:', argv[1])
        summary = pd.read_csv(argv[1])
        summary.drop(columns=['Unnamed: 0'], inplace=True)
        print(summary)
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
        summary = get_stat_summaries(training_set, 4096, run_parallel=True)
        summary.to_csv(base_dir + '/stat_summary.csv')
        print('Features have been saved to:', base_dir + '/stat_summary.csv')

    training_set = summary.values
    feature_count = training_set.shape[-1] - 1
    print(feature_count)
    print(training_set)

    # Training parameters
    batch_size=93
    epochs=2000
    
    model_name = base_dir + '/earthquake-predictions-keras-model-' + datetime.now().strftime('%Y-%m-%d_%H.%M.%S') + '-feature_count-' + str(feature_count) + '-batch_size-' + str(batch_size) + '-epochs-' + str(epochs) + '.hdf5'

    # extract(summary.iloc[:, :-1], summary.iloc[:, -1])

    print(20*'*', 'Start of training', 20*'*')
    print(20*'*', 'Keras model will be saved to:', model_name, 20*'*')
    model = Rnn(feature_count)
    model.fit(training_set, batch_size=batch_size, epochs=epochs, model_name=model_name)
    print(20*'*', 'End of training', 20*'*')

    print(20*'*', 'Start of prediction ', 20*'*')
    predict(model)
    print(20*'*', 'End of prediction ', 20*'*')
'''

if __name__ == '__main__':
    main(sys.argv)
