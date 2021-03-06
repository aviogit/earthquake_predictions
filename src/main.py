#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

import sys
import os
import gzip
import io

from visualizer import save_summary_plot
from preprocessor import get_stat_summaries
from feature_extractor import extract
from models.rnn import Rnn
from keras.models import load_model
from acoustic_graphs import plot_acoustic_signal_and_spectrum
from acoustic_graphs import load_images

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

import scipy.fftpack

from joblib import Parallel, delayed

from dataclasses import dataclass

base_dir = '/tmp/LANL-Earthquake-Prediction-train-csv-gzipped'



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


def create_labeled_test_set(config):
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
		avg_ttf_df  = pd.DataFrame(curr_ttf, index=np.arange(0, config.segment_size), columns=['time_to_failure'], dtype=np.float32)
		test_df     = pd.read_csv(base_dir + '/test/' + seg_id + '.csv')
		labeled_seg = pd.concat([test_df, avg_ttf_df], axis=1)		# we concat two columns, the TTF one with always the same value
		segs.append(labeled_seg)					# because the features extractor uses just the last TTF value

	print(f'Performing df.concat of {nsegs} segments...')			# when include_y=True is passed as parameter
	df = pd.concat(segs, ignore_index=True)
	print(df[0:2])
	print(df[-2:])

	#features = get_stat_summaries(df, config.segment_size, do_fft=True, do_stft=True, run_parallel=False, include_y=True)
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
		print('Prediction for submission no:', i, ' - id: ', seg_id, ' - time to failure:', submission.time_to_failure[i])

	milliseconds = datetime.utcnow().strftime('%f')					# these are useful for huge-batch-parallel processing of pre-trained models
	submission_name = 'submission-' + base_name + '-' + milliseconds + '.csv'
 
	submission.head()
	submission.to_csv(submission_name)


# Oh-oh-oh! Very bad news!!! calling drop_useless_features() as it is now (e.g. with all the means, etc.) totally
# wipes information from the dataset! Predictions becomes all in the range of more or less 5! This stuff is incredible!

# Just leaving the mean produces excellent results wrt removing it!
def drop_useless_features(df):
	initial_featurecount = len(df.columns)
	df.drop(columns=[ 'fft_min', 'fft_max', 'fft_min_first5k', 'fft_max_first5k', 'fft_min_last5k', 'fft_mean_first5k', 'fft_max_first1k', 'fft_trend', 'fft_trend_abs', 'abs_min', 'std_first5k', 'std_last5k', 'std_first1k', 'std_last1k', 'trend', 'trend_abs', 'hann_window_mean', 'meanA', 'varAnorm', 'skewA', 'kurtAnorm', 'meanB', 'varBnorm', 'skewB', 'kurtBnorm', 'min_roll_std10', 'change_abs_roll_std10', 'mean_roll_mean10', 'change_abs_roll_mean10', 'min_roll_std100', 'change_abs_roll_std100', 'mean_roll_mean100', 'change_abs_roll_mean100', 'min_roll_std1000', 'change_abs_roll_std1000', 'mean_roll_mean1000', 'change_abs_roll_mean1000' ], inplace=True)
	for col in df.columns:
		if "stft_" in col:
			df.drop(col, axis=1, inplace=True)

	print(f'Dropped {initial_featurecount-len(df.columns)} features, now dataset has shape: {df.shape}')
	return len(df.columns)


def drop_useless_features_below_score(df, config):
		initial_featurecount = len(df.columns)
		score_threshold = config.do_drop_useless_features_below_score
		score_fname = config.do_drop_useless_features_score_file
		scores = pd.read_csv(score_fname)
		scores.drop(columns=['Unnamed: 0'], inplace=True)
		scores = scores[scores['Value'] >= score_threshold]
		#print(scores['Feature'])
		important_features = scores['Feature'].tolist()
		#important_features.append('mean')
		important_features.append('time_to_failure')		# can you believe that I've run 10 experiments without this?
		#print(important_features)
		df = df[important_features]
		#print(df)
		print(100*'#')
		print(100*'-')
		print(100*'*')
		print(f'Dropped {initial_featurecount-len(df.columns)} features, now dataset has shape: {df.shape}')
		print(100*'*')
		print(100*'-')
		print(100*'#')
		return df, len(df.columns)-1

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
def clean_ttf_floats(df):
	factor = 10000
	# because both unique() and groupby() fail silently with my floats!
	print(f'Cleaning up dirty TTF values in the {len(df.index)}-rows long dataframe, please wait...')
	df['time_to_failure'] = (df['time_to_failure'] * factor).astype(int) / factor


def convolve_acoustic_data(df, df_name, segment_size, chip_size=4096, for_humans = False):
	clean_ttf_floats(df)

	basedir  = '/mnt/ros-data/datasets/LANL-Earthquake-Prediction/' + df_name + '-acoustic-graphs-for-conv2D-chip-size-' + str(chip_size)
	if not os.path.exists(basedir):
		print(f'Creating directory {basedir}')
		os.mkdir(basedir, 0o755)

	#Parallel(n_jobs=8*4, prefer='threads')(
	Parallel(n_jobs=8, prefer='processes')(
		delayed(plot_acoustic_signal_and_spectrum)(df, chip_size, i, basedir)
		for i in range(0, len(df), chip_size))

	'''
	for i in range(0, len(df), chip_size):
		abs_counter = int(i / chip_size)
		chip = df.iloc[i:i+chip_size, :]
		print(chip)
		print(i)
		print(len(chip))

		plot_acoustic_signal_and_spectrum(chip, abs_counter)

		print(f'Graph no. {abs_counter}')
		abs_counter += 1
	'''



def load_standard_features(feat_fname, test_set_feat_fname):
	print(f'Loading training set features from file: {feat_fname}')
	features          = pd.read_csv(feat_fname)
	if 'Unnamed: 0' in features.columns:
		features.drop(columns=['Unnamed: 0'], inplace=True)
	print(f'Loading test     set features from file: {test_set_feat_fname}')
	test_set_features = pd.read_csv(test_set_feat_fname)
	if 'Unnamed: 0' in test_set_features.columns:
		test_set_features.drop(columns=['Unnamed: 0'], inplace=True)
	if 'seg_id' in test_set_features.columns:
		test_set_features.drop(columns=['seg_id'], inplace=True)
	if 'seg_id.1' in test_set_features.columns:
		test_set_features.drop(columns=['seg_id.1'], inplace=True)
	
	print(features)
	print(test_set_features)
	feature_count          = len(features.columns)-1		# remove time_to_failure
	test_set_feature_count = len(test_set_features.columns)-1

	return features, test_set_features, feature_count, test_set_feature_count


def create_standard_features_for_training_set(fname, config):
	# Process training set
	print('Opening and reading file:', fname)
	training_set = pd.read_csv(fname, compression='gzip', dtype={'acoustic_data': np.float32, 'time_to_failure': np.float64})
	print(training_set.head())
	print(training_set.describe())

	if config.do_convolution_instead_of_manual_FE:
		convolve_acoustic_data(training_set, 'training-set', config.segment_size, chip_size)
		sys.exit(0)

	print('Extracting features from the training set...')
	features = get_stat_summaries(training_set, config.segment_size, do_fft=True, do_stft=True,
					run_parallel=config.create_features_in_parallel)

	feature_count = len(features.columns)-1

	# build the common suffix for every output file in this run
	base_name = config.base_time + '-feature_count-' + str(feature_count)

	# don't forget to save our features if we didn't loaded them before!
	features_fname = base_dir + '/training-set-features-' + base_name + '.csv'
	features.to_csv(features_fname)
	print(f'Training set features have been saved to: {features_fname}')
	# --------------------

	return features, feature_count

def create_standard_features_for_test_set(config):
	# Process test set
	labeled_test_set       = create_labeled_test_set(config)

	if config.do_convolution_instead_of_manual_FE:
		convolve_acoustic_data(labeled_test_set, 'test-set', config.segment_size, chip_size)
		sys.exit(0)

	test_set_features      = get_stat_summaries(labeled_test_set, config.segment_size, do_fft=True, do_stft=True, run_parallel=config.create_features_in_parallel, include_y=True)

	test_set_feature_count = len(test_set_features.columns)-1

	#print(test_set_features)
	#print(test_set_feature_count)

	base_name = config.base_time + '-feature_count-' + str(test_set_feature_count)

	# don't forget to save our features if we didn't loaded them before!
	test_set_features_fname = base_dir + '/test-set-features-' + base_name + '.csv'
	test_set_features.to_csv(test_set_features_fname)
	print(f'Test set features have been saved to: {test_set_features_fname}')
	# --------------------

	return test_set_features, test_set_feature_count



def show_correlation_map():
	colormap = plt.cm.RdBu
	plt.figure(figsize=(14,12))
	plt.title('Pearson Correlation of Features', y=1.05, size=15)
	sns.heatmap(training_set.astype(float).corr(),linewidths=0.1,vmax=1.0, 
		square=True, cmap=colormap, linecolor='white', annot=True)




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

	@dataclass
	class Config:
		segment_size:				int	= 150000
		do_create_standard_features:		bool	= False
		do_load_standard_features:		bool	= False
		do_train_model:				bool	= False
		do_load_model:				bool	= False
		do_differentiate_features_series:	bool	= False
		do_drop_useless_features:		bool	= False
		do_drop_useless_features_below_score:	int	= -1
		do_drop_useless_features_score_file:	str	= ''
		do_convolution_instead_of_manual_FE:	bool	= False
		do_logistic_regression:			bool	= False
		do_rescale:				bool	= False
		do_use_lgbm_model:			bool	= False
		do_use_xgboost_model:			bool	= False
		do_use_timedistributed:			bool	= False
		do_use_do_use_convlstm:			bool	= False
		do_load_signal_images:			bool	= False
		create_features_in_parallel:		bool	= False
		do_predict:				bool	= False
		chip_size:				int	= 150000
		model:					str	= 'lstm-128'
		batch_size:				int	= 32
		epochs:					int	= 1000
		base_time:				str = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

	config = Config()

	config.segment_size				= 150000

	config.do_create_standard_features		= False
	config.do_load_standard_features		= True

	config.do_train_model				= True
	config.do_load_model				= False

	config.do_differentiate_features_series		= False
	config.do_drop_useless_features			= False
	config.do_drop_useless_features_below_score	= 500
	#config.do_drop_useless_features_score_file	= base_dir + '/lgbm-all-330-feats-totally-wrong-predictions/lgbm-feature-importances.csv'
	config.do_drop_useless_features_score_file	= '/mnt/ros-data/datasets/LANL-Earthquake-Prediction/new-featureset-864/features-importance/lgbm-feature-importances.csv'
	config.do_rescale				= True			# Bring everything in the range [0, 1]

	config.do_convolution_instead_of_manual_FE	= False
	config.do_logistic_regression			= False

	#config.model					= 'lstm-128'
	#config.model					= 'lstm-64-double'
	config.model					= 'catboost'
	#config.model					= 'lgbm'
	config.do_use_lgbm_model			= False
	config.do_use_xgboost_model			= False
	config.do_use_timedistributed			= False
	config.do_use_convlstm				= False

	config.do_load_signal_images			= False

	config.create_features_in_parallel		= True

	config.do_predict				= True
	# Training parameters
	config.batch_size				= 16
	config.epochs					= 4000


	if config.do_convolution_instead_of_manual_FE:
		config.chip_size			= 150000


	# Perform "manual" features engineering (FE)
	if config.do_create_standard_features:
		fname = base_dir + '/train.csv.gz'
		#fname = base_dir + '/LANL-Earthquake-Prediction-series-no-000.csv.gz'	# uncomment this to do a quicktest before every major change
	
		features, feature_count				= create_standard_features_for_training_set	(fname, config)
		test_set_features, test_set_feature_count	= create_standard_features_for_test_set		(config)

		# These are still DataFrames
		training_set = features
		test_set     = test_set_features
	# Load features previously created with "manual" features engineering (and saved to csv)
	if config.do_load_standard_features:
		print('Loading csv files with training and test set features...')
		feat_fname		= argv[1]
		test_set_feat_fname	= argv[2]
		features, test_set_features, feature_count, test_set_feature_count = load_standard_features(feat_fname, test_set_feat_fname)

		if config.do_drop_useless_features:
			feature_count		= drop_useless_features(features)
			test_set_feature_count	= drop_useless_features(test_set_features)
		if config.do_drop_useless_features_below_score != -1:
			features, feature_count				= drop_useless_features_below_score(features, config)
			test_set_features, test_set_feature_count	= drop_useless_features_below_score(test_set_features, config)

		if config.do_differentiate_features_series:
			differentiate_features_series(features,          feature_count,          '/tmp/features')
			differentiate_features_series(test_set_features, test_set_feature_count, '/tmp/test_set_features')

		# These are still DataFrames
		training_set = features
		test_set     = test_set_features



	# Check if we loaded/created the same amount of features for training set and test set
	if feature_count != test_set_feature_count:
		print()
		print(f'Fatal error! Trying to use a training set of {feature_count} features and a test set of {test_set_feature_count} features!')
		print()
		sys.exit(0)

	# Let's redefine this for the last time because now we have feature_count == test_set_feature_count and maybe we even loaded the features
	base_name = config.base_time + '-feature_count-' + str(feature_count)

	print(f'Using a training set of {feature_count} features and {len(features)} rows.')
	print(f'Using a test     set of {test_set_feature_count} features and {len(test_set_features)} rows.')


	if len(argv) > 2:



		if config.do_convolution_instead_of_manual_FE:
			training_set_dir	= argv[1] # /mnt/ros-data/datasets/LANL-Earthquake-Prediction/training-set-acoustic-graphs-for-conv2D...
			test_set_dir		= argv[2] # /mnt/ros-data/datasets/LANL-Earthquake-Prediction/training-set-acoustic-graphs-for-conv2D...

			if len(argv) >= 2:
				config.do_load_model	= True
				model_name	= argv[3]

			if not config.do_load_model:
				training_set, labels = load_images(training_set_dir)
				print(f'Loaded {len(training_set)} images with {len(labels)} labels.')
				print(f'Training set has shape', training_set.shape)

				feature_count = 10	# bogus value
				model = Rnn(config, feature_count)
				#model.create_convolutional_model(training_set)
				model.create_conv_lstm_model(training_set)

				if config.do_use_timedistributed or config.do_use_convlstm:
					# Here we reshape the (for the moment) 102 images
					# in 2 samples of 51 image series each
					# After several hours of trial and error,
					# I discovered that also the input (and thus the
					# last Dense layer) will have shape (?, 51)
					# At least, this is the only thing that makes
					# sense right now...
	
					# looking at this post:
					# https://stackoverflow.com/questions/49432852/estimating-high-resolution-images-from-lower-ones-using-a-keras-model-based-on-c/49468183#49468183
					# the 5 dimensions needed by ConvLSTM2D layers are:
					# (samples, time, rows, cols, channels)
	
					time_grouping = 5
					n_samples = int(training_set.shape[0] / time_grouping)
	
					print(training_set.shape)
					training_set = training_set.reshape(n_samples, time_grouping, training_set.shape[1], training_set.shape[2], 1)
					print(training_set.shape)
					print(labels.shape)
					labels = labels.reshape            (n_samples, time_grouping)
					print(labels.shape)

			if not config.do_load_model:
				print(20*'*', 'Start of training', 20*'*')
				model_name           = base_dir + '/earthquake-predictions-CNN-LSTM-keras-model-'     + config.base_time + '.hdf5'
				model.cnn_lstm_fit(training_set, labels,
						batch_size=batch_size, epochs=epochs,	# only two params (except for the inner model structure)
						model_name=model_name)			# just for saving the model
				print(20*'*', 'End of training', 20*'*')
			else:
				feature_count = 10	# bogus value
				model = Rnn(config, feature_count)
				print(20*'*', 'Loading pre-trained Keras model', 20*'*')
				print(20*'*', 'Keras model will be loaded from:', model_name, 20*'*')
				model = load_model(model_name)
				model.summary()
				print(20*'*', 'End of loading', 20*'*')


			# Here we just have "tentative labels", useful for "validation cheating"
			# or trying to compute MAE on the fly...
			test_set, labels = load_images(test_set_dir)
			print(f'Loaded {len(test_set)} images with {len(labels)} labels.')
			print(f'Training set has shape', test_set.shape)

			if config.do_use_timedistributed or config.do_use_convlstm:
				time_grouping = 5
				n_samples = int(test_set.shape[0] / time_grouping)

				print(test_set.shape)
				test_set = test_set.reshape(n_samples, time_grouping, test_set.shape[1], test_set.shape[2], 1)
				print(test_set.shape)
				print(labels.shape)
				labels = labels.reshape            (n_samples, time_grouping)
				print(labels.shape)

			predictions = model.predict(test_set)
			print(20*'*', 'Start of prediction ', 20*'*')
			#predict_single_on_scaled_features(model, test_set, config.base_time)
			'''
			for i in range(len(test_set[0])):
				prediction = model.predict(test_set[i])
			'''
			prediction = model.predict(test_set)
			print(prediction)
			print(20*'*', 'End of prediction ', 20*'*')

			sys.exit()






	if config.do_load_model:
		model_name = argv[3]

		model = Rnn(config, feature_count)			# Because time_to_failure is our y!
		x_train, y_train, x_valid, y_valid = model.create_dataset(training_set, test_set)

		print(20*'*', 'Loading pre-trained Keras model', 20*'*')
		print(20*'*', 'Keras model will be loaded from:', model_name, 20*'*')
		model = load_model(model_name)
		print(20*'*', 'End of loading', 20*'*')
		model.summary()

	if config.do_train_model:
		model_name           = base_dir + '/earthquake-predictions-keras-model-'     + base_name + '.hdf5'
		scaled_features_name = base_dir + '/earthquake-predictions-scaled-features-' + base_name + '.csv'


		print(20*'*', 'Start of training', 20*'*')
		print(20*'*', 'Keras model will be saved to:', model_name, 20*'*')
		model = Rnn(config, feature_count)			# We already subtracted 1 to take into account time_to_failure

		if config.model == 'lgbm-trimmed':
			x_train, y_train, x_valid, y_valid = model.create_dataset(training_set, test_set)

			print(x_train.shape)
			print(y_train.shape)
			print(x_valid.shape)

			# Not understanding why -2, with -1 we have two 'mean' columns
			x_train_pd = pd.DataFrame(x_train[:,:-1], index=np.arange(0, len(x_train)), columns=training_set.columns[:-2], dtype=np.float64)
			y_train_pd = pd.DataFrame(y_train, index=np.arange(0, len(y_train)), columns=training_set.columns[-1:], dtype=np.float64)
			x_test_pd  = pd.DataFrame(x_valid[:,:-1], index=np.arange(0, len(x_valid)), columns=test_set.columns[:-2],     dtype=np.float64)

			'''
			print(x_train_pd)
			print(y_train_pd)
			print(x_test_pd)
			for i in x_train_pd.columns:
				print(i)
			for i in training_set.columns[:-2]:
				print(i)
			sys.exit(0)
			'''

			model.create_fit_predict_lgbm_trimmed_model(x_train_pd, y_train_pd, x_test_pd)
			sys.exit(0)

		if config.model == 'catboost':
			x_train, y_train, x_valid, y_valid = model.create_dataset(training_set, test_set)

			print(x_train.shape)
			print(y_train.shape)
			print(x_valid.shape)

			# Not understanding why -2, with -1 we have two 'mean' columns
			x_train_pd = pd.DataFrame(x_train[:,:-1], index=np.arange(0, len(x_train)), columns=training_set.columns[:-2], dtype=np.float64)
			y_train_pd = pd.DataFrame(y_train, index=np.arange(0, len(y_train)), columns=training_set.columns[-1:], dtype=np.float64)
			x_test_pd  = pd.DataFrame(x_valid[:,:-1], index=np.arange(0, len(x_valid)), columns=test_set.columns[:-2],     dtype=np.float64)
			model.create_fit_predict_catboost_model(x_train_pd, y_train_pd, x_test_pd)
			sys.exit(0)


		x_train, y_train, x_valid, y_valid = model.fit(training_set,	# we may also try to validate our model during training
				batch_size=config.batch_size,
				epochs=config.epochs,
				model_name=model_name,				# to save the model at the end of ALL the epochs (besides checkpoints)
				validation_set=test_set)

		print(20*'*', 'End of training', 20*'*')

	if config.do_predict:
		print(20*'*', 'Start of prediction ', 20*'*')
		predict_single_on_scaled_features(model, x_valid, base_name)
		print(20*'*', 'End of prediction ', 20*'*')


if __name__ == '__main__':
	main(sys.argv)
