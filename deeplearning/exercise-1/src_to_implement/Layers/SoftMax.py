import numpy as np


class SoftMax(object):
    def __init__(self):
        self.activation_values = None

    def forward(self, input_tensor):
        shifted_input = input_tensor - np.max(input_tensor)
        self.activation_values = np.exp(shifted_input)/sum(np.exp(shifted_input))
        return np.copy(self.activation_values)

    def backward(self, error_tensor):
        # We have to return En-1 from here for En
        En_1 = (self.activation_values, (error_tensor - np.dot(error_tensor)))
        return En_1