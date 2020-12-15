import numpy as np
import scipy.ndimage


class Conv(object):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.num_channels = convolution_shape[0]

        self.weights_shape = (num_kernels, *convolution_shape)

        self.weights = np.random.uniform(0, 1, self.weights_shape)
        self.bias = None

        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optmizer):
        self._optimizer = optmizer

    def forward(self, input_tensor):
        # batch * channels * height * width
        # batch * channels * height

        if type(self.stride_shape) is int:
            str_y, str_x = self.stride_shape, self.stride_shape
        else:
            if len(self.stride_shape) == 2:
                str_y, str_x = self.stride_shape
            else:
                str_y = self.stride_shape[0]

        out = None
        bias = self.bias if self.bias is not None else np.array([0])
        for img_index, image in enumerate(input_tensor):
            feature_maps = None
            for index, kernel in enumerate(self.weights):
                if len(self.convolution_shape) == 3:
                    if self.num_channels > 1:
                        aa = scipy.ndimage.convolve(
                            image, kernel, mode='constant', cval=0)[self.num_channels - 2][::str_y, ::str_x] + bias
                    else:
                        aa = scipy.ndimage.convolve(image, kernel, mode='constant', cval=0)[::str_y, ::str_x] + bias
                elif len(self.convolution_shape) == 2:
                    if self.num_channels > 1:
                        aa = scipy.ndimage.convolve(
                            image, kernel, mode='constant', cval=0)[self.num_channels - 2][::str_y] + bias
                    else:
                        aa = scipy.ndimage.convolve(image, kernel, mode='constant', cval=0)[::str_y] + bias

                if feature_maps is None:
                    feature_maps = np.zeros((self.weights.shape[0], *aa.shape))
                feature_maps[index] = aa
            if out is None:
                out = np.zeros((input_tensor.shape[0], *feature_maps.shape))
            out[img_index] = feature_maps

        return out

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize(self.weights_shape,
                                                 np.prod(self.convolution_shape),
                                                 np.prod(self.convolution_shape[1:]) * self.num_kernels)
        bias = bias_initializer.initialize((1, 1), 1, 1)

        self.weights = weights
        self.bias = bias

    def backward(self, error_tensor):
        # We have to return En-1 from here for En, gradient w.r.t to input vector
        En_1 = np.dot(self.weights[:-1], error_tensor.T)

        # Gradient of error w.r.t weights
        self._gradient_weights = np.dot(error_tensor.T, self.input_tensor).T
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        return En_1.T

