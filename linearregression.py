from model import Model
import numpy as np


class LinearRegression(Model):

    def __init__(self):
        self.theta = None
        self.epochs = 25
        self.learning_rate = 0.001
        self.trained = False

    def fit(self, x, y, epochs=25, learning_rate=0.001):
        self.validate_input(x, y, epochs)
        self.reinitialize(x.shape[1] + 1)
        self.assign_hyperparameters(epochs, learning_rate)
        x_with_bias = np.hstack((np.ones((len(x), 1)), x))
        self.train(x_with_bias, y)

    def predict(self, x):
        if self.trained is False:
            raise Exception('Model has not been trained yet')
        x_with_bias = np.hstack((np.ones((len(x), 1)), x))
        return np.dot(x_with_bias, self.theta)

    def describe(self):
        return f"Model trained: {self.trained}\n" \
               f"Epochs: {self.epochs}\n" \
               f"Learning rate: {self.learning_rate}\n" \
               f"Model parameters: {self.theta}"

    def train(self, x, y):
        for i in range(self.epochs):
            current_prediction = np.dot(x, self.theta)
            current_error = y - current_prediction
            batch_derivative = (-2 / len(x)) * np.sum(np.square(current_error))
            self.theta = self.theta - self.learning_rate * batch_derivative
        self.trained = True

    def reinitialize(self, feature_count):
        self.trained = False
        self.theta = np.zeros(feature_count)

    def assign_hyperparameters(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def validate_input(self, x_input, y_input, epochs):
        if x_input is None or y_input is None or epochs is None:
            raise Exception('Input cannot be undefined')
        if len(x_input) == 0 or len(y_input) == 0:
            raise Exception('Provided training dataset is empty')
        if len(x_input) != len(y_input):
            raise Exception('Independent variables data count has to be equal target variable data count')
        if epochs <= 0:
            raise Exception('Epochs should be greater than zero')
