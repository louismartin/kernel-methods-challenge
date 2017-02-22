import os

import matplotlib.pyplot as plt
import numpy as np


def load_images(path):
    images = np.genfromtxt(path, delimiter=",")
    # There is a trainling comma which adds one pixel, reshape to normal shape
    n_images = images.shape[0]
    n_pixels = images.shape[1] - 1
    images = images[:, :n_pixels]
    return images


def plot_image(img):
    assert img.shape == (3072,)
    # Reshape image to (3, 32, 32) then to (32, 32, 3)
    img = img.reshape((3, 32, 32))
    img = img.transpose((1, 2, 0))
    # Convert centered values to ints between 0 and 255
    img = ((img - np.min(img))/(img.max() - img.min()) * 255).astype(np.uint8)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
