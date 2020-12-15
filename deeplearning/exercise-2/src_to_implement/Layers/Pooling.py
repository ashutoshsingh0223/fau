import numpy as np


class Pooling(object):
    def __init__(self, stride_shape, pooling_shape):
        self.input_shape = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass
