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


# The basic idea for this script is to take the usual features that constitute the usual test set
# (i.e. the 2624x225 features extracted from the seg_* test set) and "paste" (or as pythonists say,
# "zip") them with the best time_to_failure column we have at the moment.
# While I'm writing these lines of comment, the "best" TTF column I have is the one coming from
# second-submissions-avg.csv with score 1.473.
# I've also produced a new "fitter" model with val_loss = 0.44984 but it has been trained on the
# previous "best TTF column" (that honestly I can't track where it came from, really).
# So it's the moment to create another "best" TTF column and try to train the "biggest working model"
# on that.
def main(argv):

	if len(argv) <= 1:
		print('Error. Please provide one feature.csv file.')
		sys.exit(0)
	if len(argv) <= 2:
		print('Error. Please provide the "best" submission file you have at the moment.')
		sys.exit(0)

	if len(argv) > 1:
		features_fname = argv[1]
		print('Loading features from file:', features_fname)
		features = pd.read_csv(argv[1])
		features.drop(columns=['Unnamed: 0'], inplace=True)
		print(features.head())

	submission_fname = argv[2]
	submission_avg   = pd.read_csv(
		submission_fname,						# Ok, let's try to read a "crafted" average submission
		#index_col='seg_id',						# (among our best two) and let's try to use TTF values
		dtype={"time_to_failure": np.float32})				# as validation set for the training...
	print(submission_avg.head())

	features['time_to_failure'] = submission_avg['time_to_failure']
	#print(submission_avg['time_to_failure'])
	print(features.head())

	features.to_csv(features_fname + '-new-best-TTF-with-' + submission_fname + '.csv')
	sys.exit(0)

if __name__ == '__main__':
    main(sys.argv)
