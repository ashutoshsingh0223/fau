import numpy as np


class L1_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        if weights.ndim == 2:
            w_1 = np.sum(np.abs(weights)) * self.alpha
        else:
            w_1 = np.sum(np.abs(weights), axis=tuple(range(weights.ndim))[1:]) * self.alpha
        return w_1


class L2_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):

        if weights.ndim == 2:
            w_2 = np.square(np.linalg.norm(weights, ord='fro')) * self.alpha
        else:
            w_2 = np.zeros((weights.shape[0],))
            for i in range(len(weights)):
                w = np.square(np.linalg.norm(weights[i])) * self.alpha
                w_2[i] = w
        return w_2
