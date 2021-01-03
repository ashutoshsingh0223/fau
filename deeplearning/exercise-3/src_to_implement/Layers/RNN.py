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
        self._gradient_weights = None
        self._optimizer = None

        self.ht_1 = None
        self.weights = None

        # self.input_tensor = None
        self.tanh_activations = None
        self.sigmoid_activations = None
        self.cocatenated_input = None

        self.input = FullyConnected(input_size + hidden_size, hidden_size)
        self._weights = self.input.weights
        self.tanh = TanH()
        self.output = FullyConnected(hidden_size, output_size)
        self.sigmoid = Sigmoid()

    def forward(self, input_tensor):
        self.tanh_activations = np.zeros((input_tensor.shape[0], self.hidden_size)).astype(np.float)
        self.sigmoid_activations = np.zeros((input_tensor.shape[0], self.output_size)).astype(np.float)
        # self.input_tensor = input_tensor.copy()
        self.cocatenated_input = np.zeros((input_tensor.shape[0], self.input_size + self.hidden_size)).astype(np.float)

        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
        for t in range(len(input_tensor)):
            # print(f'{t} - h_t_tan {self.ht_1}')
            input_vector = input_tensor[t]
            if self.ht_1 is None:
                # print('here')
                self.ht_1 = np.zeros((1, self.hidden_size))
                self._memorize = True
            input_vector = input_vector.reshape(1, -1)
            # print(f'{t} - ht_1 {self.ht_1}')
            # print(self.ht_1)
            # print(input_vector)

            input_vector = np.hstack((input_vector, self.ht_1))
            # print(f'{t} - input_vector {input_vector}')
            # print('\n')
            self.cocatenated_input[t] = input_vector
            h_t = self.input.forward(input_vector)
            # print(f'{t} - h_t {h_t}')

            h_t = self.tanh.forward(h_t)
            self.tanh_activations[t] = h_t.copy()
            # print(f'{t} - h_t_tan {h_t}')

            self.ht_1 = h_t.copy()

            y_t = self.output.forward(h_t)
            # print(f'{t} - y_t {y_t}')
            y_t = self.sigmoid.forward(y_t)
            self.sigmoid_activations[t] = y_t.copy()
            # print(f'{t} - y_t {y_t}')
            # print('\n')
            # print('\n\n')
            output_tensor[t] = y_t[0]

        if not self._memorize:
            self.ht_1 = None
        return output_tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optmizer):
        self._optimizer = optmizer

    def calculate_regularization_loss(self):
        pass

    def initialize(self, weights_init, bias_init):
        self.input.initialize(weights_init, bias_init)
        self.output.initialize(weights_init, bias_init)

    def backward(self, error_tensor):
        En_1 = np.zeros((error_tensor.shape[0], self.input_size)).astype(np.float)

        h_grad = np.zeros((1, self.hidden_size)).astype(np.float)

        gradient_weights = np.zeros(self.input.weights.shape).astype(np.float)
        total_gradient_weights_out = np.zeros(self.output.weights.shape).astype(np.float)

        for t in range(len(error_tensor)):
            # backwards counting
            t = len(error_tensor) - t - 1

            gradient = error_tensor[t].reshape(1, -1)

            # print(self.sigmoid.activations)
            self.sigmoid.activations = self.sigmoid_activations[t].reshape(1, -1)
            # print(self.sigmoid_activations[t])
            # print(self.sigmoid.activations)
            # print('\n')
            y_grad = self.sigmoid.backward(gradient)
            # print(y_grad)
            tan_activation = self.tanh_activations[t].reshape(1, -1)

            # print(self.output.input_tensor)
            self.output.input_tensor = np.hstack((tan_activation, np.array([[1.]]).astype(np.float)))
            # print(tan_activation)
            # print(self.output.input_tensor)
            # print('\n')
            y_grad = self.output.backward(y_grad)
            # print(y_grad)
            # print(self.output.weights.shape)
            # print(self.output.gradient_weights.shape)
            gradient_weights_output = self.output.gradient_weights
            # print(gradient_weights_output)
            # print(y_grad.shape, h_grad.shape)
            final = y_grad + h_grad

            # print(self.tanh.activations)
            self.tanh.activations = tan_activation
            # print(self.tanh.activations)
            # print(tan_activation)
            # print('\n')
            # print(final)final
            input_grad = self.tanh.backward(final)

            # if t == 0:
            #     h = np.zeros((1, self.hidden_size))
            # else:
            #     h = self.tanh_activations[t-1].reshape(1, -1)
            # print(self.input.input_tensor)
            self.input.input_tensor = np.hstack((self.cocatenated_input[t].reshape(1, -1), np.array([[1.]]).astype(float)))
            # print(self.input.input_tensor)

            # print('input')
            # print(input_grad)
            # print(self.input.input_tensor)
            input_grad = self.input.backward(input_grad, x=False)
            # print(input_grad)
            print('\n')
            gradient_weights_input = self.input.gradient_weights

            En_1[t] = input_grad[0][:self.input_size]
            h_grad = input_grad[0][self.input_size:].reshape(1, -1)
            gradient_weights = gradient_weights + gradient_weights_input
            # print(gradient_weights_output)
            print('\n')
            total_gradient_weights_out = total_gradient_weights_out + gradient_weights_output

        self._gradient_weights = gradient_weights
        # print(self.gradient_weights)
        print('\n\n\n')
        if self._optimizer:
            self._weights = self._optimizer.calculate_update(self._weights, gradient_weights)
            self.output.weights = self._optimizer.calculate_update(self.output.weights, total_gradient_weights_out)
            self.input.weights = self._weights
        return En_1
