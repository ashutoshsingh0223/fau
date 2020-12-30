from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, retain_probab):
        self.retain_probab = retain_probab

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor * self.retain_probab
        else:

            return input_tensor

    def backward(self, error_tensor):
        pass
