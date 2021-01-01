from abc import ABC, abstractmethod


class BaseLayer(ABC):
    def __init__(self):
        self.testing_phase = False

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError
