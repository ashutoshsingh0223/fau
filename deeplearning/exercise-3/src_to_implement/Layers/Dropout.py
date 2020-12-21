from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self):
        super.__init__()

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            # Todo: Implement inverted dropout
            return input_tensor

    def backward(self, error_tensor):
        pass
