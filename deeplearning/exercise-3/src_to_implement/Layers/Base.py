from abc import ABC, abstractmethod

import numpy as np


class BaseLayer(ABC):
    def __init__(self):
        self.testing_phase = False
        self._weights = np.zeros((2, 2))
        self._optimizer = None
        self._gradient_weights = None

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        raise NotImplementedError

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def initialize(self):
        raise NotImplementedError
