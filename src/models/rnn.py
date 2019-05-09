import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, Dropout, LSTM, Flatten, CuDNNLSTM
from keras.optimizers import adam
from keras.models import load_model

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
        self.scaler = MinMaxScaler(feature_range=(0, self.num_features))

        self.model = Sequential()
        self.model.add(CuDNNLSTM(64, input_shape=(self.num_features, 1)))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1))

        # self.model.add(Dense(units=64))
        # self.model.add(Dropout(0.1))
        # self.model.add(Dense(units=256))
        # self.model.add(Dropout(0.1))
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(Flatten())
        # self.model.add(Dense(32))
        # self.model.add(Dense(1))

        self.model.compile(optimizer=adam(lr=0.005), loss="mae")

    def fit(self, data, batch_size: int = 32, epochs: int = 20, model_name = '/tmp/keras_model.hdf5'):
        x_train, y_train = self._create_x_y(data)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        print('Saving Keras model to:', model_name)
        self.model.save(model_name)

    def predict(self, x: pd.core.frame.DataFrame):
        x_test = x.reshape(-1, self.num_features)
        scaled = self.scaler.transform(x_test)

        x_test = np.array(scaled[:, :self.num_features])
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = self.model.predict(x_test)
        return predictions[-1]

    def _create_x_y(self, data: pd.core.frame.DataFrame):
        #print(data[0:2, :self.num_features])
        x_train = self.scaler.fit_transform(data[:, :self.num_features])
        x_train, y_train = np.array(x_train[:, :self.num_features]), np.array(data[:, self.num_features])

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train
