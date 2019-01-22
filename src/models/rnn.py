import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, Dropout, LSTM, Flatten
from keras.optimizers import adam


class Rnn:
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.history = None
        self.scaler = MinMaxScaler(feature_range=(0, self.num_features))

        self.model = Sequential()
        self.model.add(CuDNNGRU(64, input_shape=(self.num_features, 1)))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1))

        self.model.compile(optimizer=adam(lr=0.005), loss="mae")

    def fit(self, data, batch_size: int = 32, epochs: int = 20):
        x_train, y_train = self._create_x_y(data)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x: pd.core.frame.DataFrame):
        x_test = x.reshape(-1, self.num_features)
        scaled = self.scaler.transform(x_test)

        x_test = np.array(scaled[:, :self.num_features])
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = self.model.predict(x_test)
        return predictions[-1]

    def _create_x_y(self, data: pd.core.frame.DataFrame):
        x_train = self.scaler.fit_transform(data[:, :self.num_features])
        x_train, y_train = np.array(x_train[:, :self.num_features]), np.array(data[:, self.num_features])

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train
