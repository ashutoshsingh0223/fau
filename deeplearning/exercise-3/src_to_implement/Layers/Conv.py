import numpy as np
import scipy.ndimage
import scipy.signal
import copy
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super(Conv, self).__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.num_channels = convolution_shape[0]

        self.weights_shape = (num_kernels, *convolution_shape)

        self._weights = np.random.uniform(0, 1, self.weights_shape)
        self.bias = np.random.uniform(0, 1, (num_kernels,))

        self._gradient_bias = None
        self._optimizer = None

        self.bias_optimizer = None
        self.input_tensor = None


    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optmizer):
        self._optimizer = copy.deepcopy(optmizer)
        self.bias_optimizer = copy.deepcopy(optmizer)

    def forward(self, input_tensor):
        # batch * channels * height * width
        # batch * channels * height
        self.input_tensor = input_tensor

        if type(self.stride_shape) is int:
            str_y, str_x = self.stride_shape, self.stride_shape
        else:
            if len(self.stride_shape) == 2:
                str_y, str_x = self.stride_shape
            else:
                str_y = self.stride_shape[0]

        out = None

        for img_index, image in enumerate(input_tensor):
            feature_maps = None
            for index, kernel in enumerate(self.weights):
                bias = self.bias[index] if len(self.bias) > 1 else self.bias
                if len(self.convolution_shape) == 3:
                    aa = scipy.ndimage.correlate(
                        image, kernel, mode='constant', cval=0)[int((self.num_channels)/2)][::str_y, ::str_x] + bias
                elif len(self.convolution_shape) == 2:
                    aa = scipy.ndimage.correlate(
                        image, kernel, mode='constant', cval=0)[int((self.num_channels)/2)][::str_y] + bias

                if feature_maps is None:
                    feature_maps = np.zeros((self.weights.shape[0], *aa.shape))
                feature_maps[index] = aa
            if out is None:
                out = np.zeros((input_tensor.shape[0], *feature_maps.shape))
            out[img_index] = feature_maps

        return out

    def initialize(self, weights_initializer, bias_initializer):
        # print(self.weights.shape)
        # print(self.bias.shape)
        weights = weights_initializer.initialize(self.weights_shape,
                                                 np.prod(self.convolution_shape),
                                                 np.prod(self.convolution_shape[1:]) * self.num_kernels)
        bias = bias_initializer.initialize((self.num_kernels, ), self.num_kernels, 1)
        # print(weights.shape)
        # print(bias.shape)

        self.weights = weights
        self.bias = bias

    def backward(self, error_tensor):
        # print(error_tensor)
        # Weights shape = (# kernels, # channels, y, x)
        # To compute En_1. Channel 1 of En_1 depends on channel 1 of all the kernels, similarly
        # channel h of En_1 depends only on the hth channel of all H kernels.
        # So transposing along dimensions 0 and 1
        axes = list(range(len(self.weights.shape)))
        axes = [1, 0] + axes[2:]
        weights = np.transpose(self.weights, axes=axes)
        # convolution shape is now (# kernels, y, x)
        convolution_shape = weights.shape[1:]

        str_x = None
        if type(self.stride_shape) is int:
            str_y, str_x = self.stride_shape, self.stride_shape
        else:
            if len(self.stride_shape) == 2:
                str_y, str_x = self.stride_shape[0], self.stride_shape[1]
            else:
                str_y = self.stride_shape[0]

        En_1 = None
        input_shape = self.input_tensor.shape

        upsampled_error_tensor = np.zeros((*error_tensor.shape[:2], *input_shape[2:]))

        for img_index, error in enumerate(error_tensor):
            feature_maps = np.zeros((weights.shape[0], *input_shape[2:]))

            upsampled_error = np.zeros((error.shape[0], *input_shape[2:]))
            if str_x is not None:
                upsampled_error[:, ::str_y, ::str_x] = error
            else:
                upsampled_error[:, ::str_y] = error

            upsampled_error_tensor[img_index] = upsampled_error
            for index, kernel in enumerate(weights):
                if len(weights.shape) > 3:
                    aa = scipy.ndimage.convolve(upsampled_error, kernel[::-1], mode='constant', cval=0)[int((self.num_kernels)/2)]
                    feature_maps[index] = aa
                else:
                    aa = scipy.ndimage.convolve(upsampled_error, kernel, mode='constant', cval=0)[
                        int((self.num_kernels) / 2)]
                    feature_maps[index] = aa

            if En_1 is None:
                En_1 = np.zeros((error_tensor.shape[0], *feature_maps.shape))
            En_1[img_index] = feature_maps

        gradient_weights = np.zeros((self.num_kernels, *self.convolution_shape))

        axes_b = list(range(len(error_tensor.shape)))
        axes_b = [0] + axes_b[2:]
        gradient_bias = np.sum(error_tensor, axis=tuple(axes_b))

        for img_index, image in enumerate(self.input_tensor):
            if len(self.convolution_shape) > 2:
                y1 = int(np.floor((self.convolution_shape[1] - 1) / 2))
                y2 = int(np.ceil((self.convolution_shape[1] - 1) / 2))
                x1 = int(np.floor((self.convolution_shape[2] - 1) / 2))
                x2 = int(np.ceil((self.convolution_shape[2] - 1) / 2))
                image = np.pad(image, ((0, 0), (y1, y2), (x1, x2)))
            else:
                y1 = int(np.floor((self.convolution_shape[1] - 1) / 2))
                y2 = int(np.ceil((self.convolution_shape[1] - 1) / 2))
                image = np.pad(image, ((0, 0), (y1, y2)))

            for channel_index, error_channel in enumerate(upsampled_error_tensor[img_index]):
                aa = scipy.signal.correlate(image, error_channel.reshape(1, *error_channel.shape), mode='valid')
                gradient_weights[channel_index] += aa

        self._gradient_weights = gradient_weights
        self._gradient_bias = gradient_bias

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return En_1

