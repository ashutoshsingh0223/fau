from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super(Flatten, self).__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)
