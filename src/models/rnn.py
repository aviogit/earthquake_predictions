import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, CuDNNGRU, Dropout, LSTM, Flatten, CuDNNLSTM

from keras.optimizers import adam
from keras.optimizers import SGD

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

class Rnn:
	def __init__(self, num_features: int):

		# This block is due to this bug: https://github.com/keras-team/keras/issues/4161#issuecomment-366031228  #
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True	# dynamically grow the memory used on the GPU
		#config.log_device_placement = True	# to log device placement (on which device the operation ran)
						# (nothing gets printed in Jupyter, only if you run it standalone)
		sess = tf.Session(config=config)
		set_session(sess)			# set this TensorFlow session as the default session for Keras
		##########################################################################################################
		'''
		2019-05-09 11:45:36.221069: E tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
		2019-05-09 11:45:36.221240: W tensorflow/core/framework/op_kernel.cc:1401] OP_REQUIRES failed at cudnn_rnn_ops.cc:1217 : Unknown: Fail to find the dnn implementation.
		Traceback (most recent call last):
		  File "./main.py", line 64, in <module>
			main()
		  File "./main.py", line 58, in main
			model.fit(training_set, batch_size=32, epochs=500)
		  File "/mnt/dropbox/dropbox/Dropbox/code/python/ml-tutorials-py/LANL-Earthquake-Prediction/earthquake_predictions/src/models/rnn.py", line 33, in fit
			self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
		  File "/mnt/ros-data/venvs/ml-tutorials/lib/python3.6/site-packages/keras/engine/training.py", line 1039, in fit
			validation_steps=validation_steps)
		  File "/mnt/ros-data/venvs/ml-tutorials/lib/python3.6/site-packages/keras/engine/training_arrays.py", line 199, in fit_loop
			outs = f(ins_batch)
		  File "/mnt/ros-data/venvs/ml-tutorials/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py", line 2715, in __call__
			return self._call(inputs)
		  File "/mnt/ros-data/venvs/ml-tutorials/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py", line 2675, in _call
			fetched = self._callable_fn(*array_vals)
		  File "/mnt/ros-data/venvs/ml-tutorials/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1439, in __call__
			run_metadata_ptr)
		  File "/mnt/ros-data/venvs/ml-tutorials/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py", line 528, in __exit__
			c_api.TF_GetCode(self.status.status))
		tensorflow.python.framework.errors_impl.UnknownError: Fail to find the dnn implementation.
			 [[{{node cu_dnnlstm_1/CudnnRNN}}]]
			 [[{{node loss/mul}}]]
		'''
		##########################################################################################################

		self.num_features = num_features
		self.history = None
		#self.scaler = MinMaxScaler(feature_range=(0, self.num_features))		# This is wrooong!
		self.scaler = MinMaxScaler(feature_range=(0, 1))

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

		# Modified original model! This one works better!
		# Ok this worked slightly better! 151 features, normalized data, batch=32, 2000 epochs and now I get 1.554!
		self.model = Sequential()
		self.model.add(CuDNNLSTM(128, input_shape=(self.num_features, 1)))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer=adam(lr=0.0001), loss="mae")


		print(self.model.summary())

	def fit(self, training_set, validation_set, batch_size: int = 32, epochs: int = 20, model_name = '/tmp/keras_model.hdf5', scaled_features_name='/tmp/scaled_features.csv'):

		x_train, y_train, x_valid, y_valid = self._create_x_y(training_set, validation_set, scaled_features_name)

		'''
		x_train, y_train = self._create_x_y(training_set)
		x_valid, y_valid = self._create_x_y(validation_set)
		'''

		'''
		print(type((x_train)), x_train.shape)
		print(type((y_train)), y_train.shape)
		print(type((x_valid)), x_valid.shape)
		print(type((y_valid)), y_valid.shape)
		'''

		#self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
		self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, batch_size=batch_size)
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
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

		# time_to_failure was not scaled before! that's why we don't do self.scaler.inverse_transform() here! :)
		predictions = self.model.predict(x_test)
		return predictions[-1]

	def _create_x_y(self, training_set: pd.core.frame.DataFrame, validation_set: pd.core.frame.DataFrame, scaled_features_name):

		print(f'Concatenating training set (shape: {training_set.shape}) and validation set (shape: {validation_set.shape}) before scaling the dataset(s).')
		train_plus_validation_set = pd.concat([training_set, validation_set], ignore_index=True)
		print(f'Concatenation complete, final dataset has shape: {train_plus_validation_set.shape}).')

		train_len = len(training_set.index)
		valid_len = len(validation_set.index)

		'''
		print(train_len, valid_len, self.num_features, training_set.columns, validation_set.columns)
		print(train_plus_validation_set)
		print(training_set.iloc[:, :self.num_features])
		print(validation_set.iloc[:, :self.num_features])
		print(train_plus_validation_set.iloc[:, :self.num_features])
		'''

		# Now we MaxMinScale (or StandardScale) on the whole dataset (train + test/validation set) because, as said on Kaggle discussions, the mean
		# of an acoustic signal doesn't have much sense (and it's probably a bias of the instrument). So we just remove it across the whole dataset.
		x_train_valid_rescaled = self.scaler.fit_transform(train_plus_validation_set.iloc[:, :self.num_features])



		'''
		labeled_training_set   = pd.concat([x_train, y_train], axis=1)
		labeled_validation_set = pd.concat([x_valid, y_valid], axis=1)
		scaled_train_valid_set = pd.concat([labeled_training_set, labeled_validation_set], ignore_index=True)
		'''

		x_train_valid_rescaled_df = pd.DataFrame(x_train_valid_rescaled,
							index=train_plus_validation_set.index,
							columns=train_plus_validation_set.columns[:-1],
							dtype=np.float32)
		print(x_train_valid_rescaled_df)
		print(f'Saving scaled features to: {scaled_features_name}')
		x_train_valid_rescaled_df.to_csv(scaled_features_name)







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






