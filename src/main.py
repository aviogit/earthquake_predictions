#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

import sys
import gzip
import io

from visualizer import save_summary_plot
from preprocessor import get_stat_summaries
from feature_extractor import extract
from models.rnn import Rnn
from keras.models import load_model

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


base_dir = '/tmp/LANL-Earthquake-Prediction-train-csv-gzipped'

# This doesn't work at all
def predict_multi(model):
	submission = pd.read_csv(
		base_dir + '/sample_submission.csv',
		index_col='seg_id',
		dtype={"time_to_failure": np.float32})

	for i, seg_id in enumerate(submission.index):
		seg = pd.read_csv(base_dir + '/test/' + seg_id + '.csv')
		summary = get_stat_summaries(seg, 4096, run_parallel=False, include_y=False)
		for j in range(summary.values.shape[0]):
			#submission.time_to_failure[i] = model.predict(summary.values.reshape(summary.values.shape[0],summary.values.shape[1],1))
			pred = model.predict(summary.values.reshape(summary.values.shape[0],summary.values.shape[1],1))		# 37x93
			X = np.arange(summary.values.shape[0]).reshape(-1, 1)
			Y = pred.reshape(-1, 1)
			print(X)
			print(Y)
			plt.scatter(X, Y) 
#                for k in range(summary.values.shape[0]):
#                    plt.scatter(k, pred[k]) 
			    #plt.plot(xs, regression_line)
			    #plt.scatter(predict_x, predict_y, s=100, color='r')

			linear_regressor = LinearRegression()	# create object for the class
			linear_regressor.fit(X, Y)		# perform linear regression
			Y_pred = linear_regressor.predict(X)	# make predictions
			plt.plot(X, Y_pred, color='red')	# draw the line into the plot

#X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
#Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
#linear_regressor = LinearRegression()  # create object for the class
#linear_regressor.fit(X, Y)  # perform linear regression
#Y_pred = linear_regressor.predict(X) # make predictions

			plt.show()

			print('Prediction for submission no.:', i, ' - id: ', seg_id, ' - feat[' + str(j) + ']', ' - time to failure:', submission.time_to_failure[i][j])

	submission.head()
	submission.to_csv('submission.csv')



def predict(model):
	submission = pd.read_csv(
		base_dir + '/sample_submission.csv',
		index_col='seg_id',
		dtype={"time_to_failure": np.float32})

	for i, seg_id in enumerate(submission.index):
		seg = pd.read_csv(base_dir + '/test/' + seg_id + '.csv')
		summary = get_stat_summaries(seg, 150000, do_fft=True, do_stft=True, run_parallel=False, include_y=False)
		submission.time_to_failure[i] = model.predict(summary.values.reshape(summary.values.shape[0],summary.values.shape[1],1))
		print('Prediction for submission no.:', i, ' - id: ', seg_id, ' - time to failure:', submission.time_to_failure[i])

	submission.head()
	submission.to_csv('submission.csv')




def main(argv):
	# 0. Read this: http://theorangeduck.com/page/neural-network-not-working?imm_mid=0f6562&cmp=em-data-na-na-newsltr_20170920

	# 0a. Visualize your data (not only with this stuff, build your own visualizers)
	# training_set = pd.read_csv('data/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
	# visualizer.plot_data(training_set)
	# preprocessor.split_sequence('data/train.csv')

	# 1. gzip the huge training set file, so to make it more manageable
	# gzip --to-stdout -1 train.csv > ~/LANL-Earthquake-Prediction-train-csv-splitted-gzip-1/train.csv.gz

	# 2. run the program without parameters to make it collect features and save them
	# ./main.py

	# 3. run the program again, and again (and again) using the saved features by just specifying the features filename
	# ./main.py /tmp/LANL-Earthquake-Prediction-train-csv-gzipped/stat_summary-parallel-now-hopefully-ok-20190509.csv | tee /tmp/LANL-Earthquake-Prediction-train-csv-gzipped/output-log-`currdate`-`currtime`-feature_count-93-batch_size-32-epochs-1000.txt

	# 4. make a quick comparison with your previous submission (if any) to see how numbers are going... (almost randomly :)
	# paste -d, submission-orig-20190508.csv submission-features-parallel-ok-20190509-174942-feature_count-93-batch_size-256-epochs-1000.csv | awk -F, '{acc+=($2-$4)*($2-$4) ; print $2-$4} END {print "----------\n"acc/NR}'

	# 5. collect the filled submission file and submit it
	# kaggle competitions submit -c LANL-Earthquake-Prediction -f /mnt/ros-data/venvs/ml-tutorials/py/LANL-Earthquake-Prediction/earthquake_predictions/src/submission-features-parallel-ok-20190509-174942-feature_count-93-batch_size-256-epochs-1000.csv -m "Another attempt with same model, same good ol' features, just larger batches and more epochs."
	# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52.8k/52.8k [00:02<00:00, 19.1kB/s]
	# Successfully submitted to LANL Earthquake Prediction

	if len(argv) > 1:
		print('Loading features from file:', argv[1])
		summary = pd.read_csv(argv[1])
		summary.drop(columns=['Unnamed: 0'], inplace=True)
		print(summary)
	else:
		fname = base_dir + '/train.csv.gz'
		#fname = base_dir + '/LANL-Earthquake-Prediction-series-no-000.csv.gz'	# remember to uncomment this to do a quicktest before every major change
		print('Opening and reading file:', fname)
		gzipped_file = gzip.open(fname, 'r')
		file_content = gzipped_file.read()

		print('Finished reading file, filling the DataFrame...')
		training_set = pd.read_csv(io.BytesIO(file_content), dtype={'acoustic_data': np.float32, 'time_to_failure': np.float64})
		#save_summary_plot(training_set)
		del file_content			# try to free some memory

		print('Extracting features...')
		summary = get_stat_summaries(training_set, 150000, do_fft=True, do_stft=True, run_parallel=True)

	training_set = summary.values
	feature_count = training_set.shape[-1] - 1
	print(feature_count)
	print(training_set)

	# Training parameters
	batch_size = 8
	epochs = 10000

	# build the common suffix for every output file in this run
	base_name = datetime.now().strftime('%Y-%m-%d_%H.%M.%S') + '-feature_count-' + str(feature_count) + '-batch_size-' + str(batch_size) + '-epochs-' + str(epochs)

	if len(argv) <= 1:
	# don't forget to save our features if we didn't loaded them before!
		summary.to_csv(base_dir + '/features-' + base_name + '.csv')
		print('Features have been saved to:', base_dir + '/stat_summary.csv')

	# extract(summary.iloc[:, :-1], summary.iloc[:, -1])

	if len(argv) > 2:
		model_name = argv[2]

		print(20*'*', 'Loading pre-trained Keras model', 20*'*')
		print(20*'*', 'Keras model will be loaded from:', model_name, 20*'*')
		model = load_model(model_name)
		print(20*'*', 'End of loading', 20*'*')
	else:
		model_name = base_dir + '/earthquake-predictions-keras-model-' + base_name + '.hdf5'

		print(20*'*', 'Start of training', 20*'*')
		print(20*'*', 'Keras model will be saved to:', model_name, 20*'*')
		model = Rnn(feature_count)
		model.fit(training_set, batch_size=batch_size, epochs=epochs, model_name=model_name)
		print(20*'*', 'End of training', 20*'*')

	print(20*'*', 'Start of prediction ', 20*'*')
	predict(model)
	print(20*'*', 'End of prediction ', 20*'*')


if __name__ == '__main__':
	main(sys.argv)
