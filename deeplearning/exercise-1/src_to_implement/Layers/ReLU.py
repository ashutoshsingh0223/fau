import numpy as np


class ReLU(object):
    def __init__(self):
        self.less_than_zero = None

    def forward(self, input_tensor):
        self.less_than_zero = input_tensor <= 0.0
        input_tensor[self.less_than_zero] = 0.0
        return input_tensor

    def backward(self, error_tensor):
        # We have to return En-1 from here for En
        En_1 = np.copy(error_tensor)
        En_1[self.less_than_zero] = 0.0
        return En_1