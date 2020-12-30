import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        self.input_shape = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_map = None

    def forward(self, input_tensor):
        # Preparing the output of the pooling operation.
        if type(self.stride_shape) is int:
            str_y, str_x = self.stride_shape, self.stride_shape
        else:
            if len(self.stride_shape) == 2:
                str_y, str_x = self.stride_shape
            else:
                str_y = self.stride_shape[0]

        # print(f'stride: {(str_y, str_x)}')
        input_shape = input_tensor.shape
        self.input_shape = input_shape
        if len(input_shape) > 3:
            y = int((input_shape[2] - self.pooling_shape[0]) / str_y)
            x = int((input_shape[3] - self.pooling_shape[1]) / str_x)
            poot_out_y_x = (y + 1, x + 1)
        else:
            y = int((input_shape[2] - self.pooling_shape[0]) / str_y)
            poot_out_y_x = (y + 1,)

        pool_out = np.zeros((*input_tensor.shape[:2], *poot_out_y_x))
        # print(pool_out[0, 0, 0, 2])

        max_map = []
        for batch_index, input_batch in enumerate(input_tensor):
            for channel_index, channel in enumerate(input_batch):
                r2 = 0
                for r in np.arange(0, input_shape[2] - self.pooling_shape[0] + 1, str_y):
                    # print(r)
                    if len(input_shape) > 3:
                        c2 = 0
                        for c in np.arange(0, input_shape[3] - self.pooling_shape[1] + 1, str_x):
                            window = input_batch[channel_index][r:r + self.pooling_shape[0], c:c + self.pooling_shape[1]]
                            pool_out[batch_index, channel_index, r2, c2] = np.max(window)
                            # print(r2, c2)
                            max_index = np.unravel_index(window.argmax(), window.shape)

                            max_map.append(((batch_index, channel_index, max_index[0] + r, max_index[1] + c),
                                           (batch_index, channel_index, r2, c2)))
                            c2 = c2 + 1
                    else:
                        window = input_batch[channel_index][r:r + self.pooling_shape[0]]
                        pool_out[channel_index, r2] = np.max(window)
                        max_index = np.unravel_index(window.argmax(), window.shape)
                        max_map.append(((batch_index, channel_index, max_index[0] + r),
                                        (batch_index, channel_index, r2)))
                    r2 = r2 + 1
        self.max_map = max_map
        # print(input_tensor)
        # print(pool_out)
        # print(self.max_map)
        return pool_out

    def backward(self, error_tensor):
        En_1 = np.zeros(self.input_shape)

        for orginal_shape, pool_shape in self.max_map:
            if len(orginal_shape) > 3:
                En_1[orginal_shape[0], orginal_shape[1], orginal_shape[2], orginal_shape[3]] += error_tensor[pool_shape[0], pool_shape[1], pool_shape[2], pool_shape[3]]

            else:
                En_1[orginal_shape[0], orginal_shape[1], orginal_shape[2]] += error_tensor[pool_shape[0], pool_shape[1], pool_shape[2]]

        return En_1
