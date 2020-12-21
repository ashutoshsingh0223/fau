import numpy as np


class Constant(object):

    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.empty(weights_shape)
        weights.fill(self.value)
        return weights.copy()


class UniformRandom(object):

    def __init__(self, lower=0, upper=1):
        self.lower = lower
        self.upper = upper

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.uniform(self.lower, self.upper, weights_shape)
        return weights.copy()


class Xavier(object):

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(np.divide(2, (fan_out + fan_in)))
        weights = np.random.normal(0, sigma, weights_shape)
        return weights.copy()


class He(object):

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(np.divide(2, fan_in))
        weights = np.random.normal(0, sigma, weights_shape)
        return weights.copy()
