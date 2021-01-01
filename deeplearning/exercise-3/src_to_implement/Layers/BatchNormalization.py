import numpy as np
from Layers.Base import BaseLayer


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super(BatchNormalization, self).__init__()
        self.gamma = None
        self.beta = None
        self.channels = channels
        self.mean = None
        self.std = None

    def forward(self, input_tensor):
        if self.channels > 1:
            mean_batch = np.mean(input_tensor)
            std_batch = np.std(input_tensor)
        else:
            mean_batch = np.mean(input_tensor, axis=(2, 3))
            std_batch = np.std(input_tensor, axis=(2, 3))

        input_tensor = np.divide(
            np.subtract(input_tensor, mean_batch),
            np.sqrt(
                np.add(
                    np.square(std_batch), np.finfo(float).eps)))
        return

    def initialize(self, weights_initializer, bias_initializer):
        self.gamma = weights_initializer.initialize((1, self.channels), 1, self.channels)
        self.beta = bias_initializer.initialize((1, self.channels), 1, self.channels)

    def backward(self, error_tensor):
        pass
