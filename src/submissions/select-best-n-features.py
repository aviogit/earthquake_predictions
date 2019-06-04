#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import sys
import gzip
import io

from datetime import datetime

from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2		# this one is only for classification problems!
#from sklearn.feature_selection import f_classif
#from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

import seaborn as sns					# for the heatmap


def exrf(train_sample, validation_sample, features, seed=42):
	log_base = np.e
	exrf_est = ExtraTreesRegressor(n_estimators=1000,
                                   criterion='mae',
                                   max_features='auto',
                                   max_depth=None,
                                   bootstrap=True,
                                   min_samples_split=4,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0,
                                   max_leaf_nodes=None,
                                   random_state=seed)

	exrf_est.fit(train_sample[features], np.log1p(train_sample['time_to_failure']) / np.log(log_base))
	prediction = exrf_est.predict(validation_sample[features])
	exrf_prob = np.power(log_base, prediction) - 1
	print(validation_sample['time_to_failure'], exrf_prob, 'EXTRA-RF')
	return exrf_prob 



def main(argv):

	if len(argv) <= 1:
		print('Error. Please provide one feature.csv file.')
		sys.exit(0)
	if len(argv) <= 2:
		print('Error. Please provide the number of "best" features to be selected.')
		sys.exit(0)

	if len(argv) > 1:
		print('Loading features from file:', argv[1])
		features = pd.read_csv(argv[1])
		features.drop(columns=['Unnamed: 0'], inplace=True)
		print(features.head())

	n_best = int(argv[2])

	X = features.iloc[:,0:-1]	#independent columns
	y = features.iloc[:,-1]		#target column i.e time to failure
	#y.reshape(1, len(y))

	'''
	print(X)
	print(y)
	'''


	#y = pd.DataFrame(y, index=X.index) #, columns=['Y'])
	scaler = MinMaxScaler(feature_range=(0, 1))
	X_scaled = scaler.fit_transform(X)
	X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
	#y = pd.DataFrame(y, index=X.index) #, columns=['Y'])
	print(y[:10])
	print(type(X_scaled), type(y))
	print(X_scaled.shape, y.shape)


	feat_train, feat_test = train_test_split(features, test_size=0.20)
	print(feat_train)
	print(feat_test)

	#apply SelectKBest class to extract top 10 best features
	#bestfeatures = SelectKBest(score_func=chi2, k=10)
	#bestfeatures = SelectKBest(score_func=f_classif, k=n_best)		# This one doesn't work, it always return the last n feature columns
	#bestfeatures = SelectKBest(score_func=f_regression, k=n_best)		# This one doesn't work either, it always returns the first n feature columns
	bestfeatures = SelectKBest(score_func=mutual_info_regression, k=n_best)
	fit = bestfeatures.fit_transform(X_scaled, y)
	print(fit)
	print(type(fit))
	#print(fit.shape)

	real_features_labels = feat_train.iloc[:, 0:-1].columns
	exrf(feat_train, feat_test, real_features_labels, seed=42)

	sys.exit(0)

	mask = bestfeatures.get_support() #list of booleans
	mask = bestfeatures.get_support(indices=True)
	new_features = [] # The list of your K best features
	print(mask)
	#print(fit.pvalues_)

	feature_names = list(X.columns.values)
	for boolean, feature in zip(mask, feature_names):
		if boolean:
			new_features.append(feature)
			print(feature, 'IS ONE OF THE BEST', n_best)
		else:
			print(feature, 'is not one of the best', n_best)
	
	bestfeats = pd.DataFrame(fit, columns=new_features)
	print(bestfeats.head())
	print("")
	print(new_features)
	print("")

	'''
	# This doesn't work either, all 150 features are inside the top_corr_features array
	#get correlations of each features in dataset
	corrmat = features.corr()
	top_corr_features = corrmat.index
	num_features = len(features.index)
	print(top_corr_features)
	print(len(top_corr_features))
	#plt.figure(figsize=(num_features, num_features), dpi=100)
	#plot heat map
	sns.set()
	plt.figure(figsize=(50, 50))
	g=sns.heatmap(features[top_corr_features].corr(),annot=True,cmap="RdYlGn")
	'''

	# feature extraction
	pca = PCA(n_components=10)
	fit = pca.fit(X.values)
	# summarize components
	print("Explained Variance: %s" % fit.explained_variance_ratio_)
	print(fit.components_)


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
