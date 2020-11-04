import numpy as np
import scipy
import matplotlib.pyplot as plt


class Checker:

    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        if self.resolution % (2 * self.tile_size) != 0:
            raise ValueError('resolution parameter should always be an even number')
        self.output = None

    def show(self):
        plt.gray()
        if self.output is None:
            board = self.draw()
        plt.imshow(self.output)

    def draw(self):
        black_tile = np.zeros((self.tile_size, self.tile_size))
        white_tile = np.full((self.tile_size, self.tile_size), 255)
        black_white = np.concatenate((black_tile, white_tile), axis=1)
        white_black = np.concatenate((white_tile, black_tile), axis=1)
        odd_column = np.tile(black_white, (1, self.resolution/(self.tile_size*2)))
        even_column = np.tile(white_black, (1, self.resolution/(self.tile_size*2)))
        double_strip = np.concatenate((odd_column, even_column), axis=0)
        board = np.tile(double_strip, (self.resolution / (self.tile_size * 2), 1))
        self.output = board
        board_copy = np.copy(board)
        return board_copy

    
class Circle():

    def __init__(self):
        pass

    def show(self):
        pass

    def draw(self):
        pass


class ColorSpectrum():

    def __init__(self):
        pass

    def show(self):
        pass

    def draw(self):
        pass