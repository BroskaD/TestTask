import time
from typing import Any
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from keras import datasets, layers, models, losses
import tensorflow as tf


class DigitClassificationInterface:

    def __init__(self):
        self.__model = None

    def predict(self, data: Any) -> int:
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class CnnClassificator(DigitClassificationInterface):

    def __init__(self):
        super().__init__()
        self.__input_shape = (28, 28, 1)
        self.__build_model()

    def __build_model(self):
        self.__model = models.Sequential([
            layers.Conv2D(6, 5, activation='tanh', padding='same', input_shape=self.__input_shape),
            layers.AveragePooling2D(2),
            layers.Activation('sigmoid'),
            layers.Conv2D(16, 5, activation='tanh'),
            layers.AveragePooling2D(2),
            layers.Activation('sigmoid'),
            layers.Conv2D(120, 5, activation='tanh'),
            layers.Flatten(),
            layers.Dense(84, activation='tanh'),
            layers.Dense(10, activation='softmax')
        ])
        self.__model.compile(optimizer='adam', loss=losses.categorical_crossentropy, metrics=['accuracy'])

    def predict(self, data) -> int:
        data = data / 255
        data = tf.expand_dims(data, axis=0, name=None)
        predicted = self.__model.predict([data])
        return np.argmax(predicted).item()

    def train(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train = tf.convert_to_tensor(x_train, np.float32) / 255
        y_train = tf.one_hot(tf.convert_to_tensor(y_train), 10)
        x_train = tf.expand_dims(x_train, axis=3, name=None)
        self.__model.fit(x_train, y_train, batch_size=64, epochs=3)


class RandomForestClassificator(DigitClassificationInterface):

    def __init__(self):
        super().__init__()
        self.__model = RandomForestClassifier()

    def predict(self, data: np.ndarray) -> int:
        data = data / 255.0
        result = self.__model.predict([data])
        return result.item()

    def train(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        row_count = x_train.shape[0]
        x_train = x_train.reshape((row_count, -1)) / 255.0
        self.__model.fit(x_train, y_train)


class RandClassificator(DigitClassificationInterface):

    def __init__(self):
        super().__init__()

    def predict(self, data: np.ndarray) -> int:
        return random.randint(0, 9)

    def train(self):
        time.sleep(2)


class DigitClassifier:

    def __init__(self, method_name: str):
        if method_name == 'cnn':
            self.__classifier = CnnClassificator()
        elif method_name == 'rf':
            self.__classifier = RandomForestClassificator()
        elif method_name == 'rand':
            self.__classifier = RandClassificator()

    def train(self):
        self.__classifier.train()

    def predict(self, data: Any) -> int:
        return self.__classifier.predict(data)


# Prepare samples for testing different cases
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
cnn_example = x_test[0]
cnn_example = tf.convert_to_tensor(cnn_example, np.float32)
cnn_example = tf.expand_dims(cnn_example, axis=2, name=None)

rf_example = x_test[0].ravel()

rand_example = x_test[0][9:19, 9:19]

# Algorithms defining and training
cnn_classifier = DigitClassifier('cnn')
cnn_classifier.train()
rf_classifier = DigitClassifier('rf')
rf_classifier.train()
rand_classifier = DigitClassifier('rand')
rand_classifier.train()

# Prediction
cnn_result = cnn_classifier.predict(cnn_example)
rf_result = rf_classifier.predict(rf_example)
rand_result = rand_classifier.predict(rand_example)

print(cnn_result, rf_result, rand_result)


