import numpy as np


class FullyConnected(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self._optimizer = None
        self.input_tensor = None
        self._gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optmizer):
        self._optimizer = optmizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        w_t_x = np.dot(self.weights.T, input_tensor.T).T
        w_t_x = w_t_x + np.ones((input_tensor.shape[0], 1))
        return np.copy(w_t_x)

    def backward(self, error_tensor):
        # We have to return En-1 from here for En, gradient w.r.t to input vector
        En_1 = np.dot(self.weights, error_tensor.T)

        # Gradient of error w.r.t weights
        self._gradient_weights = np.dot(error_tensor.T, self.input_tensor).T
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        return En_1.T

