import os

import matplotlib.pyplot as plt
import numpy as np


def load_images(path):
    """ Read a csv file from path and returns a numpy array """
    # Check if we have a .npy (numpy) version of the images (faster)
    path_npy = path.replace(".csv", ".npy")
    if os.path.exists(path_npy):
        images = np.load(path_npy)
    else:
        images = np.genfromtxt(path, delimiter=",")
        # A trailing comma adds one pixel, remove it
        n_images = images.shape[0]
        n_pixels = images.shape[1] - 1
        images = images[:, :n_pixels]
        np.save(path_npy, images)
    return images


def vec2img(X):
    """
    Takes images of shape (n_samples, 3072) or (3072,) and reshape to
    (n_samples, 32, 32 ,3)
    """
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
    assert X.shape[1] == 3*32*32
    n_samples = X.shape[0]
    X = X.reshape((n_samples, 3, 32, 32))
    X = X.transpose((0, 2, 3, 1))
    return X


def plot_image(img):
    assert img.shape == (3072,)
    img = vec2img(img)[0]
    # Convert centered values to ints between 0 and 255
    img = ((img - np.min(img))/(img.max() - img.min()) * 255).astype(np.uint8)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
