import numpy as np


class Sgd(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum(object):
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.past_gradient = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.past_gradient:
            update = np.add(np.multiply(self.momentum_rate, self.past_gradient),
                                 np.multiply(self.learning_rate, gradient_tensor))
            weight_tensor = np.subtract(weight_tensor, update)
        else:
            update = np.multiply(self.learning_rate, gradient_tensor)
            weight_tensor = np.subtract(weight_tensor, update)

        self.past_gradient = update
        return weight_tensor.copy()


class Adam(object):
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_1 = None
        self.r_1 = None
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v_1 is None:
            v = np.multiply(np.subtract(1., self.mu), gradient_tensor)
        else:
            v = np.add(np.multiply(self.mu, self.v_1),
                       np.multiply(np.subtract(1., self.mu), gradient_tensor))

        if self.r_1 is None:
            r = np.dot(np.subtract(1., self.rho) * gradient_tensor, gradient_tensor)
        else:
            r = np.add(np.multiply(self.rho, self.r_1),
                       np.dot(np.subtract(1., self.rho) * gradient_tensor, gradient_tensor))

        v_hat = v / np.add(np.subtract(1, np.power(self.mu, self.k)), np.finfo(float).eps)
        r_hat = r / np.add(np.subtract(1, np.power(self.rho, self.k)), np.finfo(float).eps)

        weight_tensor = weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps))

        self.v_1 = v
        self.r_1 = r
        self.k += 1

        return weight_tensor.copy()
