import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self._memorize = False

        self.ht_1 = None

        # self.input_tensor = None
        self.tanh_activations = None
        self.sigmoid_activations = None
        self.cocatenated_input = None
        self.output_input = None

        self.input = FullyConnected(input_size + hidden_size, hidden_size)
        self._weights = self.input.weights
        self.tanh = TanH()
        self.output = FullyConnected(hidden_size, output_size)
        self.sigmoid = Sigmoid()
        self.regu_loss = 0.0

    def calculate_regularization_loss(self, layer):
        if self._optimizer:
            if self._optimizer.regularizer:
                self.regu_loss += self._optimizer.regularizer.norm(layer.weights)

    def forward(self, input_tensor):
        self.tanh_activations = np.zeros((input_tensor.shape[0], self.hidden_size)).astype(np.float)
        self.output_input = np.zeros((input_tensor.shape[0], self.hidden_size + 1)).astype(np.float)
        self.sigmoid_activations = np.zeros((input_tensor.shape[0], self.output_size)).astype(np.float)
        self.cocatenated_input = np.zeros((input_tensor.shape[0], self.input_size + self.hidden_size + 1)).astype(np.float)

        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
        for t in range(len(input_tensor)):
            input_vector = input_tensor[t]
            if self.ht_1 is None:
                self.ht_1 = np.zeros((1, self.hidden_size))

            input_vector = input_vector.reshape(1, -1)

            input_vector = np.hstack((input_vector, self.ht_1))

            h_t = self.input.forward(input_vector)
            self.cocatenated_input[t] = self.input.input_tensor.copy()

            h_t = self.tanh.forward(h_t)

            self.tanh_activations[t] = self.tanh.activations.copy()

            self.ht_1 = h_t.copy()

            y_t = self.output.forward(h_t)
            self.output_input[t] = self.output.input_tensor.copy()

            y_t = self.sigmoid.forward(y_t)
            self.sigmoid_activations[t] = self.sigmoid.activations.copy()

            output_tensor[t] = y_t[0]
        if not self._memorize:
            self.ht_1 = None
        return output_tensor

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
        self.input.weights = self._weights

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    def initialize(self, weights_init, bias_init):
        self.input.initialize(weights_init, bias_init)
        self.output.initialize(weights_init, bias_init)
        self._weights = self.input.weights

    def backward(self, error_tensor):
        En_1 = np.zeros((error_tensor.shape[0], self.input_size)).astype(np.float)

        h_grad = np.zeros((1, self.hidden_size))

        gradient_weights = np.zeros(self.input.weights.shape).astype(np.float)
        total_gradient_weights_out = np.zeros(self.output.weights.shape).astype(np.float)

        for t in range(len(error_tensor)):
            # backwards counting
            t = len(error_tensor) - t - 1
            # print(t)
            gradient = error_tensor[t].reshape(1, -1)

            self.sigmoid.activations = self.sigmoid_activations[t].reshape(1, -1)
            # print("sig", np.array_equal(self.sigmoid.activations, self.sigmoid_activations[t].reshape(1, -1)), t)
            y_grad = self.sigmoid.backward(gradient)

            self.output.input_tensor = self.output_input[t].reshape(1, -1)
            # print("out", np.array_equal(self.output.input_tensor, self.output_input[t].reshape(1, -1)), t)
            y_grad = self.output.backward(y_grad)
            gradient_weights_output = self.output.gradient_weights
            total_gradient_weights_out = total_gradient_weights_out + gradient_weights_output

            final = y_grad + h_grad

            self.tanh.activations = self.tanh_activations[t].reshape(1, -1)
            # print("tan", np.array_equal(self.tanh.activations, self.tanh_activations[t].reshape(1, -1)), t)
            input_grad = self.tanh.backward(final)

            self.input.input_tensor = self.cocatenated_input[t].reshape(1, -1)
            # print("out", np.array_equal(self.input.input_tensor, self.cocatenated_input[t].reshape(1, -1)), t)
            input_grad = self.input.backward(input_grad)
            gradient_weights_input = self.input.gradient_weights

            En_1[t] = input_grad[0][:self.input_size]
            h_grad = input_grad[0][self.input_size:].reshape(1, -1)
            gradient_weights = gradient_weights + gradient_weights_input

        self._gradient_weights = gradient_weights

        if self._optimizer:
            self.input.weights = self._optimizer.calculate_update(self.input.weights.copy(), gradient_weights)
            self.output.weights = self._optimizer.calculate_update(self.output.weights.copy(), total_gradient_weights_out)
            self._weights = self.input.weights

        return En_1
