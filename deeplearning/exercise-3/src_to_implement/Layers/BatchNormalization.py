import numpy as np
from Layers.Base import BaseLayer


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super(BatchNormalization, self).__init__()
        self.channels = channels
        self.mean = np.zeros((1, self.channels))
        self.std = np.zeros((1, self.channels))

        self.mean_batch = np.zeros((1, self.channels))
        self.std_batch = np.zeros((1, self.channels))

        self.bias = np.zeros((1, self.channels))
        self.weights = np.ones((1, self.channels))
        self.input_shape = None
        self.input_tensor = None
        self.normalized_input = None
        self.alpha = 0.8
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optmizer):
        self._optimizer = optmizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def calculate_mean_std(self, array):
        if not self.testing_phase:
            self.mean_batch = np.mean(array, axis=0)
            self.std_batch = np.std(array, axis=0)
            self.mean = self.alpha * self.mean + self.mean_batch * (1 - self.alpha)
            self.std = self.alpha * self.std + self.std_batch * (1 - self.alpha)
            return self.mean_batch, self.std_batch
        else:
            return self.mean, self.std

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        self.input_tensor = input_tensor

        if input_tensor.ndim == 2:

            mean_b, std_b = self.calculate_mean_std(input_tensor)
            numerator = input_tensor - mean_b
            deno = np.sqrt(np.square(std_b) + np.finfo(float).eps)
            self.normalized_input = numerator / deno

            y = self.weights * self.normalized_input + self.bias

            return y
        else:
            input_tensor_transposed = np.transpose(input_tensor, axes=(0, 2, 3, 1))
            temp = input_tensor_transposed.reshape(-1, self.channels)

            mean_b, std_b = self.calculate_mean_std(temp)

            numerator = input_tensor_transposed - mean_b
            self.normalized_input = np.divide(numerator,
                                                np.sqrt(np.add(np.square(std_b), np.finfo(float).eps)))
            y = self.weights * self.normalized_input + self.bias
            y = np.transpose(y, axes=(0, 3, 1, 2))
            return y

    def reformat(self, tensor):
        if tensor.ndim == 4:
            tensor_ = np.transpose(tensor, axes=(0, 2, 3, 1))
            tensor_ = tensor_.reshape(-1, self.channels)
        else:
            # self.input_shape (#batch, #channels, #y, #x)
            # tensor_shape (#batch, #channels)
            shape = (self.input_shape[0], self.input_shape[2], self.input_shape[3], self.input_shape[1])
            tensor_ = tensor.reshape(*shape)
            tensor_ = np.transpose(tensor_, axes=(0, 3, 1, 2))
        return tensor_

    def initialize(self, x, y):
        self.bias = np.zeros((1, self.channels))
        self.weights = np.ones((1, self.channels))

    def backward(self, error_tensor):
        tensor = self.input_tensor
        ndim = error_tensor.ndim

        if ndim == 4:
            tensor = np.transpose(self.input_tensor, axes=(0, 2, 3, 1))
            error_tensor = np.transpose(error_tensor, axes=(0, 2, 3, 1))

        shape = error_tensor.shape

        grad_norm_in = (error_tensor * self.weights).reshape(-1, self.channels)
        # Already storing transposed normalized input
        gradient_weights = np.multiply(error_tensor, self.normalized_input)

        # Reshaping for common calculations
        tensor = tensor.reshape(-1, self.channels)
        grad_norm_in = grad_norm_in.reshape(-1, self.channels)
        error_tensor = error_tensor.reshape(-1, self.channels)
        gradient_weights = gradient_weights.reshape(-1, self.channels)

        var_norm = np.divide(1, np.sqrt(np.square(self.std_batch) + np.finfo(float).eps))
        mean_shift = tensor - self.mean_batch

        grad_var = np.sum(
            grad_norm_in * mean_shift * (-0.5) * np.power(var_norm, 3),
            axis=0)

        grad_mean = np.sum(grad_norm_in * -1 * var_norm, axis=0) + grad_var * np.divide(np.sum(-2 * mean_shift, axis=0), len(error_tensor))

        En_1 = grad_norm_in * var_norm + grad_var * np.divide(2 * mean_shift, len(error_tensor)) + grad_mean * np.divide(1, len(error_tensor)).reshape(1, -1)

        gradient_weights = np.sum(gradient_weights, axis=0).reshape(1, -1)
        gradient_bias = np.sum(error_tensor, axis=0).reshape(1, -1)

        self._gradient_weights = gradient_weights
        self._gradient_bias = gradient_bias

        if self._optimizer:
            self.bias = self._optimizer.calculate_update(self.bias, gradient_bias)
            self.weights = self._optimizer.calculate_update(self.weights, gradient_weights)

        if ndim == 4:
            # putting grad w.r.t input in proper shape by fist reshaping (#batch, #y, #x, #channels)
            # and then transposing to (#batch, #channels, #y, #x)
            return np.transpose(En_1.reshape(*shape), axes=(0, 3, 1, 2))
        else:
            return En_1
