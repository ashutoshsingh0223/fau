import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute dx and dy with cv2.Sobel. (2 Lines)
    Idx = cv2.Sobel(I, -1, 1, 0)
    Idy = cv2.Sobel(I, -1, 0, 1)

    # Step 2: Ixx Iyy Ixy from Idx and Idy (3 Lines)
    Ixx = np.square(Idx)
    Iyy = np.square(Idy)
    Ixy = np.multiply(Idx, Idy)

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur (5 Lines)
    A = cv2.GaussianBlur(Ixx, (3, 3), 1)
    B = cv2.GaussianBlur(Iyy, (3, 3), 1)
    C = cv2.GaussianBlur(Ixy, (3, 3), 1)

    #Step 4:  Compute the harris response with the determinant and the trace of T (see announcement) (4 lines)
    det = np.multiply(A, B) - np.square(C)
    trace = A + B
    R = det - k * np.square(trace)
    return R, A, B, C, Idx, Idy


def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended) : pad the response image to facilitate vectorization (1 line)
    # Padding with 1 since it is a 3*3 neighbourhood
    R_pad = np.pad(R, 1)

    # Step 2 (recommended) : create one image for every offset in the 3x3 neighborhood (6 lines).
    rolling = np.lib.stride_tricks.sliding_window_view(R_pad, (3, 3), axis=(0, 1), writeable=True)

    # Step 3 (recommended) : compute the greatest neighbor of every pixel (1 line)
    max_ = np.max(rolling, axis=(2, 3))

    # Step 4 (recommended) : Compute a boolean image with only all key-points set to True (1 line)
    boolean_array = np.logical_and(R >= max_, R > threshold)

    # Step 5 (recommended) : Use np.nonzero to compute the locations of the key-points from the boolean image (1 line)
    indices = np.nonzero(boolean_array)

    return indices


def detect_edges(R: np.array, edge_threshold: float = -0.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1 (recommended) : pad the response image to facilitate vectorization (1 line)
    R_pad = np.pad(R, 1)
    # Step 2 (recommended) : Calculate significant response pixels (1 line)

    # Step 3 (recommended) : create two images with the smaller x-axis and y-axis neighbors respectively (2 lines).
    # min along x-axis: minimum of each column
    min_x = np.min(R, axis=0, keepdims=True)
    # min along y-axis: minimum of each row
    min_y = np.min(R, axis=1, keepdims=True)

    # Step 4 (recommended) : Calculate pixels that are lower than either their x-axis or y-axis neighbors (1 line)
    boolean_array = np.logical_and(np.logical_or(R == min_x, R == min_y), R <= edge_threshold)

    # Step 5 (recommended) : Calculate valid edge pixels by combining significant and axis_minimal pixels (1 line)

    return boolean_array
