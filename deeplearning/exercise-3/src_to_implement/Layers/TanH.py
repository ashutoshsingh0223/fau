import numpy as np
from Layers.Base import BaseLayer


class TanH(BaseLayer):
    def __init__(self):
        super(TanH, self).__init__()
        self.activations = None

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations.copy()

    def backward(self, error_tensor):
        return np.multiply(error_tensor, np.subtract(1, np.square(self.activations)))
