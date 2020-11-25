import numpy as np


class SoftMax(object):
    def __init__(self):
        self.activation_values = None

    def forward(self, input_tensor):
        shifted_input = input_tensor - np.max(input_tensor, axis=1).reshape(-1, 1)
        self.activation_values = np.exp(shifted_input)/np.sum(np.exp(shifted_input), axis=1).reshape(-1, 1)
        return np.copy(self.activation_values)

    def backward(self, error_tensor):
        # We have to return En-1 from here for En
        element_wise_multiplication = np.multiply(error_tensor, self.activation_values)
        sum_of_each_En_y = np.sum(element_wise_multiplication, axis=1)
        after_subtraction = np.subtract(error_tensor, sum_of_each_En_y.reshape(-1, 1))
        En_1 = np.multiply(self.activation_values, after_subtraction)
        return En_1