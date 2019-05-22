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


# Ok, following this interesting discussion (https://www.kaggle.com/gpreda/lanl-earthquake-new-approach-eda) there's no point in:
# a) having a mean value of the acoustic signal (the feature should be dropped)
# b) having different mean values across both the training set and the test set (everything should be StandardScaled or MaxMinScaled together)
# c) - bonus - not being able to use the test set as validation set (at this point we have a vague idea of the values of TTF in test set, maybe we can use them to validate our model during training)
# So, we have to rework how we generate features: features must be now extracted at once on the training set and the test set, then concat'd together and the whole dataset must be scaled.
def predict_single(model):
	submission = pd.read_csv(
		base_dir + '/sample_submission.csv',
		index_col='seg_id',
		dtype={"time_to_failure": np.float32})

	for i, seg_id in enumerate(submission.index):
		seg = pd.read_csv(base_dir + '/test/' + seg_id + '.csv')
		features = get_stat_summaries(seg, 150000, do_fft=True, do_stft=True, run_parallel=False, include_y=False)
		submission.time_to_failure[i] = model.predict(features.values.reshape(features.values.shape[0],features.values.shape[1],1))
		print('Prediction for submission no.:', i, ' - id: ', seg_id, ' - time to failure:', submission.time_to_failure[i])

	submission.head()
	submission.to_csv('submission.csv')


def create_labeled_test_set():
	submission_avg = pd.read_csv(
		'./submissions/submissions-avg.csv',				# Ok, let's try to read a "crafted" average submission
		index_col='seg_id',						# (among our best two) and let's try to use TTF values
		dtype={"time_to_failure": np.float32})				# as validation set for the training...
	submission_avg.head()

	segment_size = 150000
	nsegs = 2624
	segs = []
	for i, seg_id in enumerate(submission_avg.index):
		curr_ttf = submission_avg.loc[seg_id, 'time_to_failure']
		print(f'Reading test segment {i}/{nsegs} with id {seg_id}. Using time_to_failure: {curr_ttf}')
		avg_ttf_df  = pd.DataFrame(curr_ttf, index=np.arange(0, segment_size), columns=['time_to_failure'], dtype=np.float32)
		test_df     = pd.read_csv(base_dir + '/test/' + seg_id + '.csv')
		labeled_seg = pd.concat([test_df, avg_ttf_df], axis=1)		# we concat two columns, the TTF one with always the same value
		segs.append(labeled_seg)					# because the features extractor uses just the last TTF value
	print(f'Performing df.concat of {nsegs} segments...')			# when include_y=True is passed as parameter
	df = pd.concat(segs, ignore_index=True)
	print(df[0:2])
	print(df[-2:])

	#features = get_stat_summaries(df, segment_size, do_fft=True, do_stft=True, run_parallel=False, include_y=True)
	#submission_avg.to_csv('submission_avg.csv')
	return df

def predict_single_on_scaled_features(model, x_test, base_name):
	submission = pd.read_csv(
		base_dir + '/sample_submission.csv',
		index_col='seg_id',
		dtype={"time_to_failure": np.float32})

	for i, seg_id in enumerate(submission.index):
		#seg = pd.read_csv(base_dir + '/test/' + seg_id + '.csv')
		#features = get_stat_summaries(seg, 150000, do_fft=True, do_stft=True, run_parallel=False, include_y=False)
		#submission.time_to_failure[i] = model.predict(features.values.reshape(features.values.shape[0],features.values.shape[1],1))
		submission.time_to_failure[i] = model.predict(x_test[i].reshape(1,x_test.shape[1],1))
		print('Prediction for submission no.:', i, ' - id: ', seg_id, ' - time to failure:', submission.time_to_failure[i])

	submission_name = 'submission-' + base_name + '.csv'
 
	submission.head()
	submission.to_csv(submission_name)


# Oh-oh-oh! Very bad news!!! calling drop_useless_features() as it is now (e.g. with all the means, etc.) totally
# wipes information from the dataset! Predictions becomes all in the range of more or less 5! This stuff is incredible!

# Just leaving the mean produces excellent results wrt removing it!
def drop_useless_features(df):
	#df.drop(columns=[ 'mean', 'fft_min', 'fft_max', 'fft_min_first5k', 'fft_max_first5k', 'fft_min_last5k', 'fft_mean_first5k', 'fft_max_first1k', 'fft_trend', 'fft_trend_abs', 'abs_min', 'std_first5k', 'std_last5k', 'std_first1k', 'std_last1k', 'trend', 'trend_abs', 'hann_window_mean', 'meanA', 'varAnorm', 'skewA', 'kurtAnorm', 'meanB', 'varBnorm', 'skewB', 'kurtBnorm', 'min_roll_std10', 'change_abs_roll_std10', 'mean_roll_mean10', 'change_abs_roll_mean10', 'min_roll_std100', 'change_abs_roll_std100', 'mean_roll_mean100', 'change_abs_roll_mean100', 'min_roll_std1000', 'change_abs_roll_std1000', 'mean_roll_mean1000', 'change_abs_roll_mean1000' ], inplace=True)
	df.drop(columns=[ 'fft_min', 'fft_max', 'fft_min_first5k', 'fft_max_first5k', 'fft_min_last5k', 'fft_mean_first5k', 'fft_max_first1k', 'fft_trend', 'fft_trend_abs', 'abs_min', 'std_first5k', 'std_last5k', 'std_first1k', 'std_last1k', 'trend', 'trend_abs', 'hann_window_mean', 'meanA', 'varAnorm', 'skewA', 'kurtAnorm', 'meanB', 'varBnorm', 'skewB', 'kurtBnorm', 'min_roll_std10', 'change_abs_roll_std10', 'mean_roll_mean10', 'change_abs_roll_mean10', 'min_roll_std100', 'change_abs_roll_std100', 'mean_roll_mean100', 'change_abs_roll_mean100', 'min_roll_std1000', 'change_abs_roll_std1000', 'mean_roll_mean1000', 'change_abs_roll_mean1000' ], inplace=True)
	for col in df.columns:
		if "stft_" in col:
			df.drop(col, axis=1, inplace=True)

	print(df.head)
	print(df.shape)

def differentiate_features_series(features, feature_count, fname_prefix_to_save_to=None):
		features_diff = features.iloc[ : , 1:feature_count].diff()
		features.iloc[ : , 1:feature_count] = features_diff.iloc[ : , 1:feature_count]
		features = features.iloc[1:]	# The first row is full of NaN
		print(features.head())
		if fname_prefix_to_save_to != None:
			fname = fname_prefix_to_save_to + '-' + str(len(features.index)) + 'x' + str(feature_count) + '.csv'
			features.to_csv(fname)
		'''
		test_set_features_diff = test_set_features.iloc[ : , :feature_count].diff()
		test_set_features.iloc[ : , :feature_count] = test_set_features_diff.iloc[ : , :feature_count]
		test_set_features = test_set_features.iloc[1:]
		print(test_set_features.head())
		test_set_features.to_csv('/tmp/test_set_features-diff-2624x160.csv')
		'''


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

	segment_size = 150000
	base_time = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
	do_differentiate_features_series	= False
	do_drop_useless_features		= False

	if len(argv) > 2:
		print(f'Loading training set features from file: {argv[1]}')
		features          = pd.read_csv(argv[1])
		features.drop(columns=['Unnamed: 0'], inplace=True)
		print(f'Loading test     set features from file: {argv[2]}')
		test_set_features = pd.read_csv(argv[2])
		test_set_features.drop(columns=['Unnamed: 0'], inplace=True)


		if do_drop_useless_features:
			drop_useless_features(features)
			drop_useless_features(test_set_features)

		print(features)
		print(test_set_features)
		feature_count          = len(features.columns)-1		# remove time_to_failure
		test_set_feature_count = len(test_set_features.columns)-1


		if do_differentiate_features_series:
			differentiate_features_series(features,          feature_count,          '/tmp/features')
			differentiate_features_series(test_set_features, test_set_feature_count, '/tmp/test_set_features')

		'''
		features_diff = features.iloc[ : , :feature_count].diff()
		features.iloc[ : , :feature_count] = features_diff.iloc[ : , :feature_count]
		features = features.iloc[1:]
		print(features.head())
		test_set_features_diff = test_set_features.iloc[ : , :feature_count].diff()
		test_set_features.iloc[ : , :feature_count] = test_set_features_diff.iloc[ : , :feature_count]
		test_set_features = test_set_features.iloc[1:]
		print(test_set_features.head())
		features.to_csv('/tmp/features-diff-4195x160.csv')
		test_set_features.to_csv('/tmp/test_set_features-diff-2624x160.csv')
		sys.exit(0)
		'''



	else:
		# Process training set
		fname = base_dir + '/train.csv.gz'
		#fname = base_dir + '/LANL-Earthquake-Prediction-series-no-000.csv.gz'	# remember to uncomment this to do a quicktest before every major change

		'''
		print('Opening and reading file:', fname)
		gzipped_file = gzip.open(fname, 'r')
		file_content = gzipped_file.read()

		print('Finished reading file, filling the DataFrame...')
		training_set = pd.read_csv(io.BytesIO(file_content), dtype={'acoustic_data': np.float32, 'time_to_failure': np.float64})
		#save_summary_plot(training_set)
		del file_content			# try to free some memory
		'''

		print('Opening and reading file:', fname)
		training_set = pd.read_csv(fname, compression='gzip', dtype={'acoustic_data': np.float32, 'time_to_failure': np.float64})
		print(training_set.head())

		print('Extracting features from the training set...')
		features = get_stat_summaries(training_set, segment_size , do_fft=True, do_stft=True, run_parallel=True)

		feature_count = len(features.columns)-1

		# build the common suffix for every output file in this run
		base_name = base_time + '-feature_count-' + str(feature_count)

		'''
		features.to_csv(base_dir + '/features-' + base_name + '.csv')
		print('Features have been saved to:', base_dir + '/stat_summary.csv')
		'''

		# don't forget to save our features if we didn't loaded them before!
		features_fname = base_dir + '/training-set-features-' + base_name + '.csv'
		features.to_csv(features_fname)
		print(f'Training set features have been saved to: {features_fname}')
		# --------------------




		# Process test set
		labeled_test_set       = create_labeled_test_set()
		test_set_features      = get_stat_summaries(labeled_test_set, segment_size, do_fft=True, do_stft=True, run_parallel=True, include_y=True)
	
		test_set_feature_count = len(test_set_features.columns)-1
		print(test_set_features)
		print(test_set_feature_count)
	
		#base_name = datetime.now().strftime('%Y-%m-%d_%H.%M.%S') + '-test_set_feature_count-' + str(test_set_feature_count) + '-batch_size-' + str(batch_size) + '-epochs-' + str(epochs)
		base_name = base_time + '-feature_count-' + str(test_set_feature_count)
	
		# don't forget to save our features if we didn't loaded them before!
		test_set_features_fname = base_dir + '/test-set-features-' + base_name + '.csv'
		test_set_features.to_csv(test_set_features_fname)
		print(f'Test set features have been saved to: {test_set_features_fname}')
		# --------------------
	


	if feature_count != test_set_feature_count:
		print()
		print(f'Fatal error! Trying to use a training set of {feature_count} features and a test set of {test_set_feature_count} features!')
		print()
		sys.exit(0)

	# Let's redefine this for the last time because now we have feature_count == test_set_feature_count and maybe we even loaded the features
	base_name = base_time + '-feature_count-' + str(feature_count)

	print(f'Using a training set of {feature_count} features and {len(features)} rows.')
	print(f'Using a test     set of {test_set_feature_count} features and {len(test_set_features)} rows.')

	# Training parameters
	batch_size = 16
	epochs = 4000

	'''
	# build the common suffix for every output file in this run
	base_name = datetime.now().strftime('%Y-%m-%d_%H.%M.%S') + '-feature_count-' + str(feature_count) + '-batch_size-' + str(batch_size) + '-epochs-' + str(epochs)
		# build the common suffix for every output file in this run
	base_name = datetime.now().strftime('%Y-%m-%d_%H.%M.%S') + '-feature_count-' + str(test_set_feature_count) + '-batch_size-' + str(batch_size) + '-epochs-' + str(epochs)
	'''

	#training_set = features.values
	#print(feature_count)
	#print(training_set)


	# extract(summary.iloc[:, :-1], summary.iloc[:, -1])

	# These are still DataFrames
	training_set = features
	test_set     = test_set_features

	if len(argv) > 3:
		model_name = argv[3]

		model = Rnn(feature_count)
		x_train, y_train, x_valid, y_valid = model.create_dataset(training_set, test_set)

		print(20*'*', 'Loading pre-trained Keras model', 20*'*')
		print(20*'*', 'Keras model will be loaded from:', model_name, 20*'*')
		model = load_model(model_name)
		print(20*'*', 'End of loading', 20*'*')

		# TODO: here we don't have any x_train, y_train, x_valid, y_valid
	else:
		model_name           = base_dir + '/earthquake-predictions-keras-model-'     + base_name + '.hdf5'
		scaled_features_name = base_dir + '/earthquake-predictions-scaled-features-' + base_name + '.csv'

		print(20*'*', 'Start of training', 20*'*')
		print(20*'*', 'Keras model will be saved to:', model_name, 20*'*')
		model = Rnn(feature_count)
		x_train, y_train, x_valid, y_valid = model.fit(training_set, test_set,				# we also try to validate our model during training
								batch_size=batch_size, epochs=epochs,		# only two params (except for the inner model structure)
								model_name=model_name, scaled_features_name=scaled_features_name)	# these are just for saving
		print(20*'*', 'End of training', 20*'*')
		#model = load_model('/tmp/LANL-Earthquake-Prediction-train-csv-gzipped/earthquake-predictions-keras-model-2019-05-15_17.10.05-feature_count-225-batch_size-8-epochs-1000.hdf5')

	print(20*'*', 'Start of prediction ', 20*'*')
	#predict_single(model)
	predict_single_on_scaled_features(model, x_valid, base_name)
	print(20*'*', 'End of prediction ', 20*'*')


if __name__ == '__main__':
	main(sys.argv)
