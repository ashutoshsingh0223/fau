import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, retain_probab):
        super(Dropout, self).__init__()
        self.retain_probab = retain_probab
        self.generator = np.random.RandomState(101)
        self.mask = None

    def forward(self, input_tensor):
        # if not self.testing_phase:
        #     self.mask = np.zeros((input_tensor.shape[0], input_tensor.shape[1]))
        #     activations = np.zeros((input_tensor.shape[0], input_tensor.shape[1]))
        #     for i in range(len(input_tensor)):
        #         self.mask[i] = self.generator.binomial(1, self.retain_probab, size=input_tensor[i].shape) / self.retain_probab
        #         print(self.mask[i])
        #         activations[i] = input_tensor[i] * self.mask[i]
        #         return activations

        if not self.testing_phase:
            self.mask = self.generator.binomial(1, self.retain_probab, size=input_tensor.shape) / self.retain_probab
            return input_tensor * self.mask

        return input_tensor

    def backward(self, error_tensor):

        return error_tensor * self.mask
