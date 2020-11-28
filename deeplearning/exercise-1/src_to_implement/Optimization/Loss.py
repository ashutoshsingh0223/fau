import numpy as np


class CrossEntropyLoss(object):
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        correct_label_indices = label_tensor == 1
        logit_values = input_tensor[correct_label_indices]
        adjusted_logit_values = logit_values + np.finfo(float).eps
        loss = sum(np.log(adjusted_logit_values) * -1)
        return loss

    def backward(self, label_tensor):
        return -1 * label_tensor/self.input_tensor
