import numpy as np
from typing import List, Tuple
import cv2

t_image_list = List[np.array]
t_str_list = List[str]
t_image_triplet = Tuple[np.array, np.array, np.array]


def show_images(images: t_image_list, names: t_str_list) -> None:
    """Shows one or more images at once.

    Displaying a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        names: A list of strings that will appear as the window titles for each image

    Returns:
        None
    """
    for image, name in zip(images, names):
        cv2.imshow(name, image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_images(images: t_image_list, filenames: t_str_list, **kwargs) -> None:
    """Saves one or more images at once.

    Saving a single image can be done by putting it in a list.
    If the paths have directories, they must already exist.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        filenames: A list of strings where each respective file will be created

    Returns:
        None
    """
    for image, name in zip(images, filenames):
        cv2.imwrite(name, image)


def scale_down(image: np.array) -> np.array:
    """Returns an image half the size of the original.

    Args:
        image: A numpy array with an opencv image

    Returns:
        A numpy array with an opencv image half the size of the original image
    """
    scale = 0.5
    scale_percent = 60  # percent of original size
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # cv2.imwrite('Small.png', resized)
    cv2.imshow("Resized", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return resized


def separate_channels(colored_image: np.array) -> t_image_triplet:
    """Takes an BGR color image and splits it three images.

    Args:
        colored_image: an numpy array sized [HxWxC] where the channels are in BGR (Blue, Green, Red) order

    Returns:
        A tuple with three BGR images the first one containing only the Blue channel active, the second one only the
        green, and the third one only the red.
    """
    img_height, img_width, _ = colored_image.shape
    red = np.zeros((img_height, img_width, 3))
    green = np.zeros((img_height, img_width, 3))
    blue = np.zeros((img_height, img_width, 3))

    red[:, :, 2] = colored_image[:, :, 2]
    green[:, :, 1] = colored_image[:, :, 1]
    blue[:, :, 0] = colored_image[:, :, 0]

    return blue, green, red