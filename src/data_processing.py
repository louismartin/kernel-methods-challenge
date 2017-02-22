import os

import numpy as np


def load_images(path):
    images = np.genfromtxt(path, delimiter=",")
    # There is a trainling comma which adds one pixel, reshape to normal shape
    n_images = images.shape[0]
    n_pixels = images.shape[1] - 1
    images = images[:, :n_pixels]
    return images
