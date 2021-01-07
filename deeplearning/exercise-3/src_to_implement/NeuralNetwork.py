import numpy as np
import copy
from tqdm import tqdm


class NeuralNetwork(object):
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.optimizer = optimizer
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def phase(self, layer, phase):
        layer.testing_phase = phase

    def forward(self):
        out, self.label_tensor = self.data_layer.next()
        for index in range(len(self.layers)):
            self.phase(self.layers[index], False)
            out = self.layers[index].forward(out)
        out = self.loss_layer.forward(out, self.label_tensor)
        return out

    def backward(self):
        En = self.loss_layer.backward(self.label_tensor)
        last_index = len(self.layers) - 1
        for index in range(len(self.layers)):
            layer = self.layers[last_index - index]
            if layer.optimizer:
                if layer.optimizer.regularizer:
                    regu_loss = layer.optimizer.regularizer.norm(layer.weights)
                    if En.ndim == 2:
                        En = En + regu_loss
                    else:
                        if En.ndim == 3:
                            axes = (0, 2, 1)
                            En = np.transpose(En, axes=axes) + regu_loss
                            En = np.transpose(En, axes=axes)
                        else:
                            En = np.transpose(En, axes=(0, 2, 3, 1)) + regu_loss
                            En = np.transpose(En, axes=(0, 3, 1, 2))
            En = layer.backward(En)

    def append_trainable_layer(self, layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for iter in tqdm(range(iterations)):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        for index in range(len(self.layers)):
            self.phase(self.layers[index], True)
            input_tensor = self.layers[index].forward(input_tensor)
        return input_tensor
