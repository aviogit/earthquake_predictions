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
	if len(argv) <= 2:
		print('Error. Please provide at least two submission.csv files')
		sys.exit(0)

	idx = 0
	submissions = [None] * (len(argv)-1)

	for tok in argv:
		if tok == argv[0]:
			continue
		print(f'Reading argv: {tok}')
		submissions[idx] = pd.read_csv(
			tok,
			index_col='seg_id',
			dtype={"time_to_failure": np.float32})
		
		#submissions[idx].reset_index().plot(kind='line', y='time_to_failure', use_index=True, ax=ax, sharex=True)
		#submissions.plot(kind='line',x='name',y='num_pets', color='red', ax=ax)
		idx += 1

	df = pd.concat(submissions)
	by_row_index = df.groupby(df.index)
	df_means = by_row_index.mean()

	print("Writing avg submissions values:\n" + str(df_means.head()) + "\nto /tmp/submissions-avg.csv")
	df_means.to_csv('/tmp/submissions-avg.csv')

	#abs_error_df(submissions, 0, 1)		# 1.564 vs. 1.578
	sys.exit(0)


if __name__ == '__main__':
    main(sys.argv)
