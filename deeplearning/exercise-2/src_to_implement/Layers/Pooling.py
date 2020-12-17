import numpy as np


class Pooling(object):
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

        max_map = {}
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
                            max_map[f'{batch_index}_{channel_index}_{r2}_{c2}'] = (max_index[0] + r, max_index[1] + c)
                            c2 = c2 + 1
                    else:
                        window = input_batch[channel_index][r:r + self.pooling_shape[0]]
                        pool_out[channel_index, r2] = np.max(window)
                        max_index = np.unravel_index(window.argmax(), window.shape)
                        max_map[f'{batch_index}_{channel_index}_{r2}_{c2}'] = (max_index[0] + r)
                    r2 = r2 + 1
        self.max_map = max_map
        return pool_out

    def backward(self, error_tensor):

        if len(error_tensor.shape) > 3:
            y, x = error_tensor.shape[2], error_tensor.shape[3]
        else:
            y = error_tensor.shape[2]

        En_1 = np.zeros(self.input_shape)
        for batch_index, error in enumerate(error_tensor):
            for channel_index, channel in enumerate(error):
                if len(error_tensor.shape) > 3:
                    for y_index, x_index in zip(range(y), range(x)):
                        index_tuple = self.max_map[f'{batch_index}_{channel_index}_{y_index}_{x_index}']
                        En_1[batch_index, channel_index, index_tuple[0], index_tuple[1]] += error[channel_index][y_index, x_index]

                else:
                    for y_index in zip(range(y)):
                        index_tuple = self.max_map[f'{batch_index}_{channel_index}_{y_index}']
                        En_1[batch_index, channel_index, index_tuple[0]] += error[channel_index][y_index]

        return En_1
