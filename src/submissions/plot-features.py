#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import sys

'''
dir='/tmp/LANL-Earthquake-Prediction-train-csv-gzipped'

cat earthquake-predictions-scaled-features-2019-05-17_14.28.45-feature_count-225.csv  | awk -F\, '{print $1" "$2}' | tail -6819 | gnuplot -p -e "plot '<cat' with lines"
cat features-2019-05-15_15.52.59-feature_count-225-batch_size-32-epochs-2000.csv  | awk -F\, '{print $1" "$2}' | tail -4195 | gnuplot -p -e "plot '<cat' with lines"
cat test_set_features-2019-05-16_17.10.36-test_set_feature_count-225-batch_size-8-epochs-10000.csv  | awk -F\, '{print $1" "$2}' | tail -2624 | gnuplot -p -e "plot '<cat' with lines"
'''



'''
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
'''

def main(argv):

	basedir = '/tmp/LANL-Earthquake-Prediction-train-csv-gzipped/'

	#scaled_features_train_test_fname = basedir + 'earthquake-predictions-scaled-features-2019-05-17_14.28.45-feature_count-225.csv'
	scaled_features_train_test_fname = basedir + 'earthquake-predictions-scaled-features-2019-05-17_16.59.52-feature_count-225.csv'
	unscaled_features_train_fname    = basedir + 'features-2019-05-15_15.52.59-feature_count-225-batch_size-32-epochs-2000.csv'
	unscaled_features_test_fname     = basedir + 'test_set_features-2019-05-16_17.10.36-test_set_feature_count-225-batch_size-8-epochs-10000.csv'

	scaled_features_train_test       = pd.read_csv(scaled_features_train_test_fname)
	unscaled_features_train          = pd.read_csv(unscaled_features_train_fname)
	unscaled_features_test           = pd.read_csv(unscaled_features_test_fname)

	scaled_features_train_test.drop(columns=['Unnamed: 0'], inplace=True)
	unscaled_features_train.drop(columns=['Unnamed: 0'], inplace=True)
	unscaled_features_test.drop(columns=['Unnamed: 0'], inplace=True)

	scaled_nfeat       = len(scaled_features_train_test.columns)
	unscaled_xtr_nfeat = len(unscaled_features_train.columns) - 1
	unscaled_xte_nfeat = len(unscaled_features_test.columns)  - 1
	
	unscaled_x = pd.concat([unscaled_features_train, unscaled_features_test], ignore_index=True)

	if scaled_nfeat != unscaled_xtr_nfeat or unscaled_xtr_nfeat != unscaled_xte_nfeat:
		print(f"Features mismatch: {scaled_nfeat} vs. {unscaled_xtr_nfeat} vs. {unscaled_xte_nfeat}")
		sys.exit(1)

	good_old_features_train_fname    = basedir + 'features-2019-05-13_20.12.31-feature_count-151-batch_size-32-epochs-1000.csv'
	good_old_features_train          = pd.read_csv(good_old_features_train_fname)
	#unscaled_x			 = good_old_features_train
	#unk_features_train_fname         = basedir + 'training-set-features-2019-05-17_16.59.52-feature_count-225.csv'
	#unscaled_x                       = pd.read_csv(unk_features_train_fname)
	#df = scaled_features_train_test
	df = unscaled_x
	for col in range(scaled_nfeat):
		ax = plt.gca()
		ax.set_xlabel("Test Sample")
		ax.set_ylabel("Value")
		ax.legend(df.columns[col])
		df.iloc[ : , col].plot(kind='line', ax=ax, sharex=True, legend=True)
		ax = plt.gca()
		good_old_features_train.iloc[ : , -1].plot(kind='line', ax=ax, sharex=True, legend=True)
		#print(df.columns[col]);
		mng = plt.get_current_fig_manager()
		#mng.window.state('withdrawn')
		mng.resize(*mng.window.maxsize())
		plt.grid(True)
		plt.show()
		
	sys.exit(0)

	if len(argv) <= 1:
		print('Error. Please provide features.csv file')
		sys.exit(0)

	features = pd.read_csv(argv[1])
	features.drop(columns=['Unnamed: 0'], inplace=True)

	num_features = features.shape[1]
	print(features)
	print(num_features)
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_features = scaler.fit_transform(features)
	features = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)

	'''
	norm_scaler = Normalizer()
	norm_features = norm_scaler.fit_transform(features)
	features = pd.DataFrame(norm_features, index=features.index, columns=features.columns)
	'''

	print(features)

	ttf_idx = features.columns.get_loc('time_to_failure')

	#fig = plt.figure()
	#ax = plt.gca()
	fig, axes = plt.subplots(nrows=2, ncols=3)
	#features.reset_index().plot(kind='line', y=[features.columns[i], 'time_to_failure'], use_index=True, subplots=True, layout=(3, 2), sharex=True, sharey=False, legend=True)
	for i in range(6):
		#fig.add_subplot(2, int(i/3)+1, (i%3)+1)
		#plt.imshow(img[i], cmap=cm.Greys_r)
		#print(features.iloc[0, :])
		#print(features.iloc[0, :]['mean'])
		print(i/3, i%3)
		#features.reset_index().plot(kind='line', y=[features.columns[i], 'time_to_failure'], use_index=True, subplots=True, ax=axes[int(i/3), i%3])
		# Nice, see: https://stackoverflow.com/questions/45985877/slicing-multiple-column-ranges-from-a-dataframe-using-iloc
		new_df = features.iloc[:, np.r_[i, ttf_idx]]
		new_df.plot(kind='line', use_index=True, ax=axes[int(i/3), i%3])
		#features.reset_index().plot(kind='line', y='time_to_failure', use_index=True) #, ax=ax)

	plt.show()

	sys.exit(0)


if __name__ == '__main__':
	main(sys.argv)
