import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()
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