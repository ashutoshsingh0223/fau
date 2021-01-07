import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._weights = np.random.uniform(0, 1, (input_size+1, output_size))
        self.input_tensor = None

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((weights, bias))

    def forward(self, input_tensor):
        x_0 = np.ones((1, input_tensor.shape[0]))
        self.input_tensor = np.hstack((input_tensor, x_0.T))
        w_t_x = np.dot(self.weights.T, self.input_tensor.T).T
        return np.copy(w_t_x)

    def backward(self, error_tensor):
        # We have to return En-1 from here for En, gradient w.r.t to input vector
        En_1 = np.dot(error_tensor, self.weights[:-1].T)
        # print(f'e-shape: {En_1.shape}')
        # Gradient of error w.r.t weights
        self._gradient_weights = np.dot(error_tensor.T, self.input_tensor).T

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        # print(f'e-shape-return: {En_1.T.shape}')
        return En_1

