import sys
import os

from datetime import datetime
from datetime import date

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D

from keras.layers import Activation, AveragePooling3D, CuDNNGRU, CuDNNLSTM, Dense, Dropout, Flatten, Lambda, LSTM, MaxPooling1D, MaxPooling2D, Reshape, TimeDistributed

from keras.optimizers import adam
from keras.optimizers import SGD

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import lightgbm as lgb
import xgboost  as xgb


import scipy.stats
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from catboost import Pool, CatBoostRegressor

# Plot CNN architecture
import pydot
from keras.utils import plot_model
from IPython.display import SVG
import keras.utils.vis_utils 
from keras.utils.vis_utils import model_to_dot

import matplotlib.pyplot as plt
import seaborn as sns

def print_tensor_shape(x, string=''):
	print(string, x.shape)
	return x
def add_lambda_print(model, name, debug=False):
	if debug:
		layer_name = re.sub(r' ', '_', name)
		model.add(Lambda(print_tensor_shape, name=layer_name, arguments={'string':name}))

class Rnn:
	def __init__(self, config, num_features=None):

		self.config = config

		# This block is due to this bug: https://github.com/keras-team/keras/issues/4161#issuecomment-366031228  #
		tf_config = tf.ConfigProto()
		tf_config.gpu_options.allow_growth = True	# dynamically grow the memory used on the GPU
		#tf_config.log_device_placement = True		# to log device placement (on which device the operation ran)
								# (nothing gets printed in Jupyter, only if you run it standalone)
		sess = tf.Session(config=tf_config)
		set_session(sess)				# set this TensorFlow session as the default session for Keras

		self.history = None
		#self.scaler = MinMaxScaler(feature_range=(0, self.num_features))	# This looked wrong initially, but it performs mostly ok
		self.scaler = MinMaxScaler(feature_range=(0, 1))

		self.num_features = num_features

		if self.config.model == 'lstm-32':
			self.create_model_lstm_32(num_features)
		if self.config.model == 'lstm-64-double':
			self.create_model_lstm_64_double(num_features)
		if self.config.model == 'lstm-128':
			self.create_model_lstm_128(num_features)
		if self.config.model == 'lstm-512':
			self.create_model_lstm_512(num_features)
		if self.config.model == 'lstm-865':
			self.create_model_lstm_865(num_features)
		if self.config.model == 'gru':
			self.create_model_gru(num_features)
		if self.config.model == 'lgbm':
			self.create_model_lgbm()
		if self.config.model == 'lgbm-trimmed':
			# this is the last day of competition, this model is an all-in-one taken from a Kaggle kernel
			pass
		if self.config.model == 'catboost':
			# this is the last day of competition, this model is an all-in-one taken from a Kaggle kernel
			pass
		if self.config.model == 'xgboost':
			self.create_model_xgboost()
		
	def create_model_lstm_32(self, num_features: int):
		print(f'Creating a new LSTM model with {self.num_features} features and 32 neurons as input layer...')
		self.model = Sequential()
		self.model.add(CuDNNLSTM(32, input_shape=(self.num_features, 1)))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.0001), loss="mae")
		print(self.model.summary())


	def create_model_lstm_64_double(self, num_features: int):
		print(f'Creating a new LSTM model with {self.num_features} features and 64 neurons as input layer...')
		self.model = Sequential()
		self.model.add(CuDNNLSTM(64, input_shape=(self.num_features, 1), return_sequences=True))
		self.model.add(Dropout(0.5))
		self.model.add(CuDNNLSTM(64))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.0001), loss="mae")
		print(self.model.summary())


	def create_model_lstm_128(self, num_features: int):
		print(f'Creating a new LSTM model with {self.num_features} features and 128 neurons as input layer...')

		# Modified original model! This one works better!
		# Ok this worked slightly better! 151 features, normalized data, batch=32, 2000 epochs and now I get 1.554!

		# Ok, this one worked well, but it still overfits too early (around 1000 epochs, maybe it has just too many neurons?)

		# Used this one again with validation set with TTF values obtained from best submission (second-submissions-avg.csv)
		# The model was so fit for validation set that the public score has been the same as second-submissions-avg.csv :D:D:D
		self.model = Sequential()
		self.model.add(CuDNNLSTM(128, input_shape=(self.num_features, 1)))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.0001), loss="mae")
		print(self.model.summary())

	def create_model_lstm_512(self, num_features: int):
		print(f'Creating a new LSTM model with {self.num_features} features and 512 neurons with dropout as input layer...')

		self.model = Sequential()
		self.model.add(CuDNNLSTM(512, input_shape=(self.num_features, 1)))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.00001), loss="mae")
		print(self.model.summary())

	def create_model_lstm_865(self, num_features: int):
		print(f'Creating a new LSTM model with {self.num_features} features and 865 neurons with dropout as input layer...')

		self.model = Sequential()
		self.model.add(CuDNNLSTM(865, input_shape=(self.num_features, 1)))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.00001), loss="mae")
		print(self.model.summary())

	def create_model_gru(self, num_features: int):
		print(f'Creating a new GRU model with {self.num_features} features and 128 neurons as input layer...')
		self.model = Sequential()
		self.model.add(CuDNNGRU(128, input_shape=(self.num_features, 1)))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.01), loss="mae")
		print(self.model.summary())

	def create_model_lgbm(self):
		'''
		params = {'num_leaves': 256,
			'min_data_in_leaf': 500,
			'objective':'regression',
			'max_depth': 32,
			'max_bin': 1024,
			'num_iterations': 10240,
			'learning_rate': 0.001,
			#"boosting": "gbdt",
			"boosting": "dart",
			"feature_fraction": 0.91,
			"bagging_freq": 1,
			"bagging_fraction": 0.91,
			"bagging_seed": 42,
			"metric": 'mae',
			"lambda_l1": 0.1,
			"verbosity": 4,
			"random_state": 42}
		'''
		params = {'num_leaves': 21,
			'min_data_in_leaf': 20,
			'objective':'regression',
			'learning_rate': 0.001,
			'max_depth': 108,
			"boosting": "gbdt",
			"feature_fraction": 0.91,
			"bagging_freq": 1,
			"bagging_fraction": 0.91,
			"bagging_seed": 42,
			"metric": 'mae',
			"lambda_l1": 0.1,
			"verbosity": 10,
			"random_state": 42}

		print(f'Creating a new LGBM model with a lot of params...')
		self.model = lgb.LGBMRegressor(**params, n_estimators=60000, n_jobs=-1)

	def create_model_xgboost(self):
		print(f'Creating a new XGBoost model with a lot of params...')
		self.model = xgb.XGBRegressor(
				learning_rate = 0.001,
				n_estimators = 2000,
				max_depth = 10,
				min_child_weight = 10,
				#gamma = 1,
				gamma = 0.9,
				subsample = 0.8,
				colsample_bytree = 0.8,
				#objective = 'binary:logistic',
				objective='reg:linear',
				#objective='reg:squarederror',
				#objective='reg:gamma', # OMG, horrible!
				#objective='reg:tweedie', # less horrible, but still horrible
				nthread = -1,
				random_state = 42,
				#eval_metric = 'mae',
				scale_pos_weight = 1)
		#.fit(x_train, y_train)
		#predictions = gbm.predict(x_test)





	def create_fit_predict_lgbm_trimmed_model(self, scaled_train_X, train_y, scaled_test_X):
		print(f'Creating a new LGBM trimmed model with a lot of params...')
		params = {'num_leaves': 21,
			'min_data_in_leaf': 20,
			'objective':'regression',
			'max_depth': 108,
			'learning_rate': 0.001,
			"boosting": "gbdt",
			"feature_fraction": 0.91,
			"bagging_freq": 1,
			"bagging_fraction": 0.91,
			"bagging_seed": 42,
			"metric": 'mae',
			"lambda_l1": 0.1,
			"verbosity": -1,
			"random_state": 42}


		base_dir = '/tmp/LANL-Earthquake-Prediction-train-csv-gzipped'

		maes = []
		rmses = []
		tr_maes = []
		tr_rmses = []
		submission = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'), index_col='seg_id')

		'''
		scaled_train_X = pd.read_csv(r'pk8/scaled_train_X_8.csv')
		df = pd.read_csv(r'pk8/scaled_train_X_8_slope.csv')
		scaled_train_X = scaled_train_X.join(df)

		scaled_test_X = pd.read_csv(r'pk8/scaled_test_X_8.csv')
		df = pd.read_csv(r'pk8/scaled_test_X_8_slope.csv')
		scaled_test_X = scaled_test_X.join(df)
		'''

		pcol = []
		pcor = []
		pval = []
		'''
		y = pd.read_csv(r'pk8/train_y_8.csv')['time_to_failure'].values
		'''
		y = train_y.copy()['time_to_failure']

		for col in scaled_train_X.columns:
			'''
			print(scaled_train_X[col])
			print(y)
			print(scaled_train_X[col].shape)
			print(y.shape)
			print(f'col: {col}')
			'''
			pcol.append(col)
			pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
			pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))

		df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
		df.sort_values(by=['cor', 'pval'], inplace=True)
		df.dropna(inplace=True)
		df = df.loc[df['pval'] <= 0.05]
		df.to_csv('pearsonr-cor.csv')

		drop_cols = []

		for col in scaled_train_X.columns:
			if col not in df['col'].tolist():
				drop_cols.append(col)

		scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
		scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

		'''
		train_y = pd.read_csv(r'pk8/train_y_8.csv')
		'''
		predictions = np.zeros(len(scaled_test_X))
		preds_train = np.zeros(len(scaled_train_X))

		print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)

		n_fold = 6
		folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)

		for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
			print(f'Working on fold {fold_} with trn_idx = {trn_idx} and val_idx = {val_idx}')
			strLog = "fold {}".format(fold_)
			print(strLog)

			X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
			y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

			local_model = lgb.LGBMRegressor(**params, n_estimators=60000, n_jobs=-1)
			local_model.fit(X_tr, y_tr,
				      eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
				      verbose=1000, early_stopping_rounds=200)

			# model = xgb.XGBRegressor(n_estimators=1000,
			#                                learning_rate=0.1,
			#                                max_depth=6,
			#                                subsample=0.9,
			#                                colsample_bytree=0.67,
			#                                reg_lambda=1.0, # seems best within 0.5 of 2.0
			#                                # gamma=1,
			#                                random_state=777+fold_,
			#                                n_jobs=12,
			#                                verbosity=2)
			# model.fit(X_tr, y_tr)

			# predictions
			preds = local_model.predict(scaled_test_X)  #, num_iteration=model.best_iteration_)
			predictions += preds / folds.n_splits
			preds = local_model.predict(scaled_train_X)  #, num_iteration=model.best_iteration_)
			preds_train += preds / folds.n_splits

			preds = local_model.predict(X_val)  #, num_iteration=model.best_iteration_)

			# mean absolute error
			mae = mean_absolute_error(y_val, preds)
			print('MAE: %.6f' % mae)
			maes.append(mae)

			# root mean squared error
			rmse = mean_squared_error(y_val, preds)
			print('RMSE: %.6f' % rmse)
			rmses.append(rmse)

			# training for over fit
			preds = local_model.predict(X_tr)  #, num_iteration=model.best_iteration_)

			mae = mean_absolute_error(y_tr, preds)
			print('Tr MAE: %.6f' % mae)
			tr_maes.append(mae)

			rmse = mean_squared_error(y_tr, preds)
			print('Tr RMSE: %.6f' % rmse)
			tr_rmses.append(rmse)

		print('MAEs', maes)
		print('MAE mean: %.6f' % np.mean(maes))
		print('RMSEs', rmses)
		print('RMSE mean: %.6f' % np.mean(rmses))

		print('Tr MAEs', tr_maes)
		print('Tr MAE mean: %.6f' % np.mean(tr_maes))
		print('Tr RMSEs', rmses)
		print('Tr RMSE mean: %.6f' % np.mean(tr_rmses))

		submission.time_to_failure = predictions
		submission.to_csv('submission_xgb_slope_pearson_6fold.csv')  # index needed, it is seg id

		pr_tr = pd.DataFrame(data=preds_train, columns=['time_to_failure'], index=range(0, preds_train.shape[0]))
		pr_tr.to_csv(r'preds_tr_xgb_slope_pearson_6fold.csv', index=False)
		print('Train shape: {}, Test shape: {}, Y shape: {}'.format(scaled_train_X.shape, scaled_test_X.shape, train_y.shape))
 
# do this in the IDE, call the function above
# if __name__ == "__main__":
#     lgb_trimmed_model()



	def create_fit_predict_catboost_model(self, train_X, train_y, test_X):
		train_columns = train_X.columns
		n_fold = 5
		folds = KFold(n_splits=n_fold, shuffle = True, random_state=42)
		
		oof = np.zeros(len(train_X))
		train_score = []
		fold_idxs = []
		# if PREDICTION: 
		predictions = np.zeros(len(test_X))
		
		feature_importance_df = pd.DataFrame()
		#run model
		for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X,train_y.values)):
			strLog = "fold {}".format(fold_)
			print(strLog)
			fold_idxs.append(val_idx)

			X_tr, X_val = train_X[train_columns].iloc[trn_idx], train_X[train_columns].iloc[val_idx]
			y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

			model = CatBoostRegressor(n_estimators=25000, verbose=-1, objective="MAE", loss_function="MAE", boosting_type="Ordered", task_type="GPU")
			model.fit(X_tr, 
				  y_tr, 
				  eval_set=[(X_val, y_val)], 
#					   eval_metric='mae',
				  verbose=2500, 
				  early_stopping_rounds=500)
			oof[val_idx] = model.predict(X_val)

			#feature importance
			fold_importance_df = pd.DataFrame()
			fold_importance_df["Feature"] = train_columns
			fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]
			fold_importance_df["fold"] = fold_ + 1
			feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
			#predictions
#			 if PREDICTION:

			predictions += model.predict(test_X[train_columns]) / folds.n_splits
			train_score.append(model.best_score_['learn']["MAE"])

		cv_score = mean_absolute_error(train_y, oof)
		print(f"After {n_fold} test_CV = {cv_score:.3f} | train_CV = {np.mean(train_score):.3f} | {cv_score-np.mean(train_score):.3f}", end=" ")

		today = str(date.today())
		#submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')

		base_dir = '/tmp/LANL-Earthquake-Prediction-train-csv-gzipped'
		submission = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))

		submission["time_to_failure"] = predictions
		submission.to_csv(f'CatBoost_{today}_test_{cv_score:.3f}_train_{np.mean(train_score):.3f}.csv', index=False)
		print(submission)







	# Ok, now we have to separate the init of the class and the model creation, because we can later recreate the model
	# after we dropped some features (e.g. 111 out 159 :) by running LogisticRegression
	def create_model(self, num_features: int):

#		self.model = Sequential()
#		self.model.add(CuDNNLSTM(96, input_shape=(self.num_features, 1)))
#		self.model.add(Dense(96, activation='relu'))
#		self.model.add(Dense(1))
#		self.model.compile(optimizer=adam(lr=0.005), loss="mae")

		'''
		self.model = Sequential()
		self.model.add(CuDNNLSTM(93, input_shape=(self.num_features, 1), return_sequences=True))
		self.model.add(CuDNNLSTM(93, return_sequences=False))
		self.model.add(Dense(93, activation='relu'))
		self.model.add(Dense(1, activation='linear'))
		self.model.compile(optimizer=adam(lr=0.001), loss='mse',  metrics = ['mse'])
		'''

		'''
		self.model = Sequential()
		self.model.add(CuDNNLSTM(64, input_shape=(self.num_features, 1), return_sequences=True))
		self.model.add(CuDNNLSTM(64, return_sequences=False))
		self.model.add(Dense(32, activation='relu'))
		self.model.add(Dense(1, activation='linear'))
		self.model.compile(optimizer=adam(lr=0.001), loss='mse',  metrics = ['mse'])
		'''


		'''
		# This model doesn't learn!!!
		self.model = Sequential()
		self.model.add(CuDNNLSTM(128, input_shape=(self.num_features, 1), return_sequences=True))
		self.model.add(Dropout(0.2))
		self.model.add(CuDNNLSTM(64, return_sequences=False))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(32, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1, activation='linear'))
		sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
		#self.model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
		self.model.compile(optimizer=sgd, loss='mse', metrics = ['mse'])
		'''



		# self.model.add(Dense(units=64))
		# self.model.add(Dropout(0.1))
		# self.model.add(Dense(units=256))
		# self.model.add(Dropout(0.1))
		# self.model.add(Dense(128, activation='relu'))
		# self.model.add(Flatten())
		# self.model.add(Dense(32))
		# self.model.add(Dense(1))


		'''
		# Original model! This one MUST work!
		# Ok this works (obviously :) but with normalized data now I get 1.578 vs. 1.564 of the original submission with un-normalized data
		self.model = Sequential()
		self.model.add(CuDNNLSTM(64, input_shape=(self.num_features, 1)))
		self.model.add(Dense(32, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.005), loss="mae")
		'''

		'''
		Slightly worse than the original model 1.602 (with 151 features) vs. 1.564 with just 93 features)
		self.model = Sequential()
		self.model.add(CuDNNLSTM(64, input_shape=(self.num_features, 1)))
		self.model.add(Dense(32, activation='relu'))
		self.model.add(Dense(1))
		sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
		self.model.compile(optimizer=sgd, loss='mae', metrics = ['mae'])
		'''

		'''
		# Modified original model! This one works better!
		# Ok this worked slightly better! 151 features, normalized data, batch=32, 2000 epochs and now I get 1.554!

		# Ok, this one worked well, but it still overfits too early (around 1000 epochs, maybe it has just too many neurons?)

		# Used this one again with validation set with TTF values obtained from best submission (second-submissions-avg.csv)
		# The model was so fit for validation set that the public score has been the same as second-submissions-avg.csv :D:D:D
		self.model = Sequential()
		self.model.add(CuDNNLSTM(128, input_shape=(self.num_features, 1)))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.0001), loss="mae")
		'''

		'''
		# This is a crap that learns too slowly
		self.model = Sequential()
		self.model.add(CuDNNLSTM(16, input_shape=(self.num_features, 1), return_sequences=True))
		self.model.add(Dropout(0.2))
		self.model.add(CuDNNLSTM(8, return_sequences=False))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(4, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.01), loss="mae")
		'''

		'''
		# This one stops learning too early as well
		self.model = Sequential()
		self.model.add(CuDNNLSTM(16, input_shape=(self.num_features, 1), return_sequences=False))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(8, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.01), loss="mae")
		'''

		'''
		# Original model again, let's see how it performs with differentiated + scaled features
		self.model = Sequential()
		self.model.add(CuDNNLSTM(64, input_shape=(self.num_features, 1)))
		self.model.add(Dense(32, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.005), loss="mae")
		'''

		'''
		self.model = Sequential()
		self.model.add(CuDNNGRU(128, input_shape=(self.num_features, 1)))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.01), loss="mae")

		print(self.model.summary())
		'''

	def create_dataset(self, training_set, validation_set):

		x_train, y_train, x_valid, y_valid = self._create_x_y(training_set, validation_set)
		return x_train, y_train, x_valid, y_valid





	def cnn_lstm_fit(self, X, y, batch_size: int = 32, epochs: int = 20, model_name = '/tmp/keras_model.hdf5'):

		# checkpoint
		filepath	="/tmp/lanl-checkpoint-{epoch:02d}-{loss:.5f}.hdf5"
		#checkpoint_vloss= ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		checkpoint_loss	= ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		#earlystopping	= EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto', baseline=0.5) #, restore_best_weights=True)
		#callbacks_list	= [checkpoint_vloss, earlystopping]

		#reduce_lr_vloss	= ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)
		reduce_lr_loss	= ReduceLROnPlateau(monitor='loss',	factor=0.2, patience=10, min_lr=0.0001)

		#callbacks_list	= [checkpoint_loss, checkpoint_vloss, reduce_lr_loss, reduce_lr_vloss]
		callbacks_list	= [checkpoint_loss, reduce_lr_loss]

		#self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
		#self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, batch_size=batch_size)
		#self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), callbacks=callbacks_list, epochs=epochs, batch_size=batch_size)

		self.model.fit(X, y, callbacks=callbacks_list, epochs=epochs, batch_size=batch_size)
		print(f'Saving Keras model to: {model_name}')
		self.model.save(model_name)

		return X, y










	def fit(self, training_set, batch_size: int = 32, epochs: int = 20,
		model_name = '/tmp/keras_model.hdf5', create_x_y_dataset=True, validation_set=None):

		if create_x_y_dataset:
			#x_train, y_train, x_valid, y_valid = self._create_x_y_normalized_across_all_data(training_set, validation_set)
			x_train, y_train, x_valid, y_valid = self._create_x_y(training_set, validation_set)

		# checkpoint
		if validation_set is None:
			filepath	="/tmp/lanl-checkpoint-{epoch:02d}-{loss:.5f}.hdf5"
		else:
			filepath	="/tmp/lanl-checkpoint-{epoch:02d}-{loss:.5f}-{val_loss:.5f}.hdf5"
		checkpoint_vloss= ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		checkpoint_loss	= ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		#earlystopping	= EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto', baseline=0.5) #, restore_best_weights=True)
		#callbacks_list	= [checkpoint_vloss, earlystopping]

		reduce_lr_vloss	= ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)
		reduce_lr_loss	= ReduceLROnPlateau(monitor='loss',	factor=0.2, patience=10, min_lr=0.0001)

		if validation_set is None:
			callbacks_list	= [checkpoint_loss, reduce_lr_loss]
		else:
			callbacks_list	= [checkpoint_loss, checkpoint_vloss, reduce_lr_loss, reduce_lr_vloss]

		#self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
		#self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, batch_size=batch_size)
		if self.config.do_use_lgbm_model or self.config.do_use_xgboost_model:
			print(80*'*', 'These are the shape of training and validation sets before training.')
			print(x_train.shape)
			print(y_train.shape)
			print(x_valid.shape)
			print(y_valid.shape)
			print(80*'*')
			eval_set = [(x_valid, y_valid)]
			self.model.fit(x_train, y_train, eval_set=eval_set, eval_metric='mae', verbose=5, early_stopping_rounds=2000)
			if self.config.do_use_lgbm_model:
				self.plot_lgbm_feature_importance(training_set)
			#self.model.show_correlation_map(training_set)
			return x_train, y_train, x_valid, y_valid


		#self.model.fit(x_train, y_train, callbacks=callbacks_list, epochs=epochs, batch_size=batch_size)
		self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), callbacks=callbacks_list, epochs=epochs, batch_size=batch_size)
		#self.model.fit(x_train, y_train, callbacks=callbacks_list)
		print(f'Saving Keras model to: {model_name}')
		self.model.save(model_name)

		return x_train, y_train, x_valid, y_valid

	def predict(self, x: pd.core.frame.DataFrame):
		'''
		x_test = x.reshape(-1, self.num_features)
		scaled = self.scaler.transform(x_test)
		'''
		scaled = x.reshape(-1, self.num_features)

		x_test = np.array(scaled[:, :self.num_features])
		if not self.config.do_use_lgbm_model and not self.config.do_use_xgboost_model:
			# Now we just have to reshape x_ sets to "add one dimension" so to make Keras happy :)
			x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

		# time_to_failure was not scaled before! that's why we don't do self.scaler.inverse_transform() here! :)

		predictions = self.model.predict(x_test)
		print('Raw predictions: ', predictions)
		return predictions[-1]

	def _create_x_y(self, training_set: pd.core.frame.DataFrame, validation_set: pd.core.frame.DataFrame):

		if 'seg_id' in validation_set.columns:
			validation_set.drop(columns=['seg_id'], inplace=True)

		train_len = len(training_set.index)
		valid_len = len(validation_set.index)

		x_valid   = None
		y_valid   = None

		if self.config.do_rescale:
			print(f'Performing MinMaxScale across all the {self.num_features} features')
			# Now we MinMaxScale (or StandardScale) the two separate datasets (train and test/validation set) because the two dataset have very different
			# behavior and properties and it makes no sense to try to handle them as "an unique thing".
			x_train_rescaled = self.scaler.fit_transform(training_set.iloc[:  , :self.num_features])
			if valid_len != 0:
				x_valid_rescaled = self.scaler.transform    (validation_set.iloc[:, :self.num_features])
			# Both x_train_rescaled and x_valid_rescaled are without their 'time_to_failure' column!

			# The training set now is: the rescaled set, up to train_len rows and up to total columns - 1 (the time_to_failure)
			x_train = np.array(x_train_rescaled)
			print(f'y index will be: {self.num_features} - column: {training_set.columns[self.num_features]}')
			y_train = np.array(training_set.iloc[: , self.num_features])

			# In a similar way, the validation set now is: the rescaled set, up to valid_len rows and up to total columns - 1 (the time_to_failure)
			if valid_len != 0:
				x_valid = np.array(x_valid_rescaled) 
				y_valid = np.array(validation_set.iloc[:, self.num_features])
		else:
			# The training set now is: the rescaled set, up to train_len rows and up to total columns - 1 (the time_to_failure)
			x_train = np.array(training_set.iloc[: , :self.num_features])
			y_train = np.array(training_set.iloc[: ,  self.num_features])

			# In a similar way, the validation set now is: the rescaled set, up to valid_len rows and up to total columns - 1 (the time_to_failure)
			x_valid = np.array(validation_set.iloc[:, :self.num_features]) 
			if valid_len != 0:
				y_valid = np.array(validation_set.iloc[:,  self.num_features])

		print(x_train.shape)
		if valid_len != 0:
			print(x_valid.shape)

		print(training_set.columns  [:self.num_features])
		if valid_len != 0:
			print(validation_set.columns[:self.num_features])


		if self.config.do_logistic_regression:
			rescaled_train_df = pd.DataFrame(x_train, index=training_set.index  , columns=training_set.columns  [:self.num_features])
			rescaled_valid_df = pd.DataFrame(x_valid, index=validation_set.index, columns=validation_set.columns[:self.num_features])
	
			cols = [ 0, 2, 29, 30, 32, 35, 38, 41, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 87, 88, 89, 90, 91, 97, 98, 99, 100, 101, 107, 109, 120, 124, 125, 136, 140, 141, 152, 156, 157 ]
	
			print('Keeping only columns:', cols)
			str_cols = ''
			for col in cols:
				str_cols += training_set.columns[col] + ' '
			print('Keeping only columns:', str_cols)
			rescaled_train_df_good_feat = rescaled_train_df.iloc[ : , cols]
			rescaled_valid_df_good_feat = rescaled_valid_df.iloc[ : , cols]
			print(rescaled_train_df_good_feat.head())
			print(rescaled_valid_df_good_feat.head())
	
			# Let's do it one more time :(
			x_train = np.array(rescaled_train_df_good_feat.iloc[:, :self.num_features])
	
			# Let's do it one more time :(
			x_valid = np.array(rescaled_valid_df_good_feat.iloc[:, :self.num_features])
	
			print(x_train)
			print(x_valid)
			print(x_train.shape)
			print(x_valid.shape)

			self.create_model(len(cols))

		if not self.config.do_use_lgbm_model and not self.config.do_use_xgboost_model and not self.config.model == 'lgbm-trimmed' and not self.config.model == 'catboost':
			# Now we just have to reshape x_ sets to "add one dimension" so to make Keras happy :)
			x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
			if valid_len != 0:
				x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

		'''
		y_train = np.reshape(y_train, (y_train.shape[0], 1))
		y_valid = np.reshape(y_valid, (y_valid.shape[0], 1))
		'''
		if self.config.do_use_xgboost_model:
			y_train = y_train.reshape((-1,1))
			y_valid = y_valid.reshape((-1,1))


		print(f'x_train.shape: {x_train.shape}')
		if valid_len != 0:
			print(f'x_valid.shape: {x_valid.shape}')

		print(f'y_train.shape: {y_train.shape}')
		if valid_len != 0:
			print(f'y_valid.shape: {y_valid.shape}')

		print(50*'-', 'Training set')
		print(x_train[:10, :])
		print('y_train:', y_train)
		if valid_len != 0:
			print(50*'-', 'Validation set')
			print(x_valid[:10, :])
			print('y_valid:', y_valid)
		print(80*'-')

		return x_train, y_train, x_valid, y_valid






	def _create_x_y_normalized_across_all_data(self, training_set: pd.core.frame.DataFrame, validation_set: pd.core.frame.DataFrame):

		print(f'Concatenating training set (shape: {training_set.shape}) and validation set (shape: {validation_set.shape}) before scaling the dataset(s).')
		train_plus_validation_set = pd.concat([training_set, validation_set], ignore_index=True)
		print(f'Concatenation complete, final dataset has shape: {train_plus_validation_set.shape}).')

		train_len = len(training_set.index)
		valid_len = len(validation_set.index)


		# Now we MinMaxScale (or StandardScale) on the whole dataset (train + test/validation set) because, as said on Kaggle discussions, the mean
		# of an acoustic signal doesn't have much sense (and it's probably a bias of the instrument). So we just remove it across the whole dataset.
		x_train_valid_rescaled = self.scaler.fit_transform(train_plus_validation_set.iloc[:, :self.num_features])


		x_train_valid_rescaled_df = pd.DataFrame(x_train_valid_rescaled,
							index=train_plus_validation_set.index,
							columns=train_plus_validation_set.columns[:-1],
							dtype=np.float32)
		print(x_train_valid_rescaled_df)
		#print(f'Saving scaled features to: {scaled_features_name}')
		#x_train_valid_rescaled_df.to_csv(scaled_features_name)


		# The training set now is: the rescaled set, up to train_len rows and up to total columns - 1 (the time_to_failure)
		x_train = np.array(x_train_valid_rescaled[:train_len, :self.num_features])
		y_train = np.array(train_plus_validation_set.iloc[:train_len, self.num_features])

		# In a similar way, the validation set now is: the rescaled set, up to valid_len rows and up to total columns - 1 (the time_to_failure)
		x_valid = np.array(x_train_valid_rescaled[:valid_len, :self.num_features]) 
		y_valid = np.array(train_plus_validation_set.iloc[:valid_len, self.num_features])

		# Now we just have to reshape x_ sets to "add one dimension" so to make Keras happy :)
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
		x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

		return x_train, y_train, x_valid, y_valid

	def plot_lgbm_feature_importance(self, training_set):
		clf = self.model
		X   = training_set

		# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
		feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X.columns)), columns=['Value','Feature'])
		feature_imp.to_csv('/tmp/lgbm-feature-importances.csv')
		
		plt.figure(figsize=(20, 10))
		sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
		plt.title('LightGBM Features (avg over folds)')
		plt.tight_layout()
		#plt.show()
		plt.savefig('/tmp/lgbm-feature-importances.png')

	def show_correlation_map(self, training_set):
		colormap = plt.cm.RdBu
		plt.figure(figsize=(14,12))
		plt.title('Pearson Correlation of Features', y=1.05, size=15)
		sns.heatmap(training_set.astype(float).corr(),linewidths=0.1,vmax=1.0, 
			square=True, cmap=colormap, linecolor='white', annot=True)

	def create_convolutional_model(self, X):
		self.model = Sequential()
		
		if self.config.do_use_timedistributed:
			#1st conv layer
			self.model.add(TimeDistributed(Conv2D(16, (7,7), padding="valid",
						input_shape=(X.shape[1],X.shape[2],1),data_format="channels_last",
						name ='conv_2d_layer_1'),
					input_shape=(51, X.shape[1], X.shape[2], 1)))
			self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
			self.model.add(BatchNormalization())
			self.model.add(Activation("relu"))
			self.model.add(Dropout(0.25))
			
			#2nd conv layer
			self.model.add(TimeDistributed(Conv2D(32, (3,3), padding="valid", name ='conv_2d_layer_2')))
			self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
			self.model.add(BatchNormalization())
			self.model.add(Activation("relu"))
			self.model.add(Dropout(0.25))
			
			#3rd conv layer
			self.model.add(TimeDistributed(Conv2D(64, (3,3), padding="valid", name ='conv_2d_layer_3')))
			self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
			self.model.add(BatchNormalization())
			self.model.add(Activation("relu"))
			self.model.add(Dropout(0.25))
			
			#4th conv layer
			self.model.add(TimeDistributed(Conv2D(128, (3,3), padding="valid", name ='conv_2d_layer_4')))
			self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
			self.model.add(BatchNormalization())
			self.model.add(Activation("relu"))
			self.model.add(Dropout(0.25))
			#self.model.add(MaxPooling2D())
		else:
			#1st conv layer
			self.model.add(Conv2D(16, (7,7), padding="valid",
						input_shape=(X.shape[1],X.shape[2],1),data_format="channels_last",
						name ='conv_2d_layer_1'))
			self.model.add(MaxPooling2D(pool_size=(2, 2)))
			self.model.add(BatchNormalization())
			self.model.add(Activation("relu"))
			self.model.add(Dropout(0.25))
			
			#2nd conv layer
			self.model.add(Conv2D(32, (3,3), padding="valid", name ='conv_2d_layer_2'))
			self.model.add(MaxPooling2D(pool_size=(2, 2)))
			self.model.add(BatchNormalization())
			self.model.add(Activation("relu"))
			self.model.add(Dropout(0.25))
			
			#3rd conv layer
			self.model.add(Conv2D(64, (3,3), padding="valid", name ='conv_2d_layer_3'))
			self.model.add(MaxPooling2D(pool_size=(2, 2)))
			self.model.add(BatchNormalization())
			self.model.add(Activation("relu"))
			self.model.add(Dropout(0.25))
			
			#4th conv layer
			self.model.add(Conv2D(128, (3,3), padding="valid", name ='conv_2d_layer_4'))
			self.model.add(MaxPooling2D(pool_size=(2, 2)))
			self.model.add(BatchNormalization())
			self.model.add(Activation("relu"))
			self.model.add(Dropout(0.25))
			#self.model.add(MaxPooling2D())
			
	
		'''	
		#FC1
		self.model.add(Dense(128))
		self.model.add(BatchNormalization())
		self.model.add(Activation("relu"))
		self.model.add(Dropout(0.25))
		
		#FC2
		#self.model.add(Dense(100,name ='feature_dense'))
		self.model.add(Dense(128, name ='feature_dense'))
		#self.model.load_weights(by_name=True,filepath = filepath)
		self.model.add(BatchNormalization())
		self.model.add(Activation("relu"))
		
		#output FC
		self.model.add(Dense(2))
		self.model.add(Activation('sigmoid'))
		#adam = optimizers.Adam(lr=0.01)
		'''
		
		if self.config.do_use_timedistributed:
			self.model.add(TimeDistributed(Flatten()))
			#self.model.add(MaxPooling2D(pool_size=(16,16)))
			#self.model.compile(loss='binary_crossentropy', metrics=[auc], optimizer='adam')
			#self.model.add(Flatten())
	
			#self.model.add(CuDNNLSTM(128, input_shape=(self.num_features, 1)))
			self.model.add(CuDNNLSTM(128, input_shape=(12, 51), name ='lstm_layer_1', return_sequences=False))	# Best model so far...
			self.model.add(Dense(64, activation='relu', name ='dense_layer_1'))
			self.model.add(Dense(1, name ='output_layer_antani'))
		else:
			#self.model.add(MaxPooling1D(pool_size=2,strides=None, padding='valid',input_shape=(50,1)))
			#self.model.add(MaxPooling2D(pool_size=(2, 2)))
			#self.model.add(Flatten())
			#self.model.add(MaxPooling2D(pool_size=(16,16)))
			#self.model.compile(loss='binary_crossentropy', metrics=[auc], optimizer='adam')
			#self.model.add(Flatten())
	
			#self.model.add(CuDNNLSTM(128, input_shape=(self.num_features, 1)))
			self.model.add(CuDNNLSTM(128, input_shape=([12]), name ='lstm_layer_1', return_sequences=False))	# Best model so far...
			self.model.add(Dense(64, activation='relu', name ='dense_layer_1'))
			self.model.add(Dense(1, name ='output_layer_antani'))


		self.model.compile(optimizer=adam(lr=0.0001), loss="mae")

		self.model.summary()
		plot_model(self.model, to_file='/tmp/cnn-model.png')
		SVG(model_to_dot(self.model).create(prog='dot', format='svg'))


	def create_conv_lstm_model(self, X, time_grouping=5):
		'''
		# We create a layer which take as input movies of shape
		# (n_frames, width, height, channels) and returns a movie
		# of identical shape.
		
		seq = Sequential()
		seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
		                   input_shape=(None, 40, 40, 1),
		                   padding='same', return_sequences=True))
		seq.add(BatchNormalization())
		
		seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
		                   padding='same', return_sequences=True))
		seq.add(BatchNormalization())
		
		seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
		                   padding='same', return_sequences=True))
		seq.add(BatchNormalization())
		
		seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
		                   padding='same', return_sequences=True))
		seq.add(BatchNormalization())
		
		seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
		               activation='sigmoid',
		               padding='same', data_format='channels_last'))
		seq.compile(loss='binary_crossentropy', optimizer='adadelta')
		'''

		print(f'Creating ConvLSTM2D model for input shape: {X.shape}')
		debug = True

		# looking at this post:
		# https://stackoverflow.com/questions/49432852/estimating-high-resolution-images-from-lower-ones-using-a-keras-model-based-on-c/49468183#49468183
		# the 5 dimensions needed by ConvLSTM2D layers are:
		# (samples, time, rows, cols, channels)


		# If going OOM, whatch out for both the number of filters
		# applied to the first convolutional layer and the batch_size
		# 2019-05-28 16:24:58.805472: W tensorflow/core/framework/op_kernel.cc:1401] OP_REQUIRES failed at conv_ops.cc:446 : Resource exhausted: OOM when allocating tensor with shape[16,16,1440,1920] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc


		n_samples = int(X.shape[0] / time_grouping)	# ah-ah! unused!
		n_filters = 4

		self.model = Sequential()
		self.model.add(ConvLSTM2D(filters=n_filters, kernel_size=(7, 7),
		                   #input_shape=(None, 40, 40, 1),
		                   input_shape=(time_grouping, X.shape[1], X.shape[2], 1),
		                   padding='same', return_sequences=True))
		add_lambda_print(self.model, 'ConvLSTM2D-1', debug)
		self.model.add(BatchNormalization())
		
		self.model.add(ConvLSTM2D(filters=n_filters, kernel_size=(3, 3),
				padding='same', return_sequences=True))
		add_lambda_print(self.model, 'ConvLSTM2D-2', debug)
		self.model.add(BatchNormalization())
		
		self.model.add(ConvLSTM2D(filters=n_filters, kernel_size=(3, 3),
				padding='same', return_sequences=True))
		add_lambda_print(self.model, 'ConvLSTM2D-3', debug)
		self.model.add(BatchNormalization())
		
		self.model.add(ConvLSTM2D(filters=n_filters, kernel_size=(3, 3),
				padding='same', return_sequences=True))
		add_lambda_print(self.model, 'ConvLSTM2D-4', debug)
		self.model.add(BatchNormalization())
		
		self.model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
				activation='sigmoid',
				padding='same', data_format='channels_last'))
		add_lambda_print(self.model, 'Conv3D', debug)

		self.model.add(AveragePooling3D((1, X.shape[1], X.shape[2])))
		add_lambda_print(self.model, 'AveragePooling3D', debug)
		self.model.add(Reshape((-1, time_grouping)))
		add_lambda_print(self.model, 'Reshape', debug)
		self.model.add(Flatten())
		add_lambda_print(self.model, 'Flatten', debug)
		self.model.add(Dense(time_grouping, name ='dense_layer_wide', activation='linear'))
		add_lambda_print(self.model, 'Dense Wide', debug)
#		self.model.add(Dense(1, name ='dense_layer_1', activation='linear'))
#		add_lambda_print(self.model, 'Dense 1', debug)



		'''
		self.model.add(Dense(
			units=1,
			name ='dense_layer_1',
			activation='sigmoid'))
		'''


		#self.model.add(Dense(64, activation='relu', name ='dense_layer_1'))
		#self.model.add(Dense(1, name ='output_layer_antani'))

		self.model.compile(optimizer=adam(lr=0.0001), loss="mae")

		self.model.summary()
		plot_model(self.model, to_file='/tmp/cnn-model.png')
		SVG(model_to_dot(self.model).create(prog='dot', format='svg'))
