from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def describe(self):
        pass

