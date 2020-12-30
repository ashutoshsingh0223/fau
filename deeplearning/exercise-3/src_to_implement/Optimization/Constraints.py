import numpy as np


class L1_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        w_1 = np.sum(np.abs(weights)) * self.alpha
        return w_1


class L2_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        w_2 = np.square(np.linalg.norm(weights, ord='fro')) * self.alpha
        return w_2
