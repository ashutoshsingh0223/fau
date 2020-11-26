import numpy as np
import copy


class NeuralNetwork(object):
    def __init__(self, optimizer):
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.optimizer = optimizer

    def forward(self):
        out, label_tensor = self.data_layer.next()
        for index in range(len(self.layers)):
            out = self.layers[index].forward(out)
        out = self.loss_layer.forward(out, label_tensor)
        return out

    def backward(self):
        _, label_tensor = self.data_layer.next()
        En = self.loss_layer.backward(label_tensor)
        last_index = len(self.layers) - 1
        for index in range(len(self.layers)):
            En = self.layers[last_index - index].backward(En)

    def append_trainable_layer(self, layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for iter in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        for index in range(len(self.layers)):
            input_tensor = self.layers[index].forward(input_tensor)
        return input_tensor
