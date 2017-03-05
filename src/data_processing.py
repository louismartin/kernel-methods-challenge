import os

import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import AffineTransform
from skimage import transform as tf


def load_images(path):
    """ Read a csv file from path and returns a numpy array """
    # Check if we have a .npy (numpy) version of the images (faster)
    path_npy = path.replace(".csv", ".npy")
    if os.path.exists(path_npy):
        images = np.load(path_npy)
    else:
        images = np.genfromtxt(path, delimiter=",")
        # A trailing comma adds one pixel, remove it
        n_pixels = images.shape[1] - 1
        images = images[:, :n_pixels]
        np.save(path_npy, images)
    return images


def vec2img(X):
    """
    Takes images of shape (n_samples, n_pixels) or (n_pixels,) and reshape to
    (n_samples, width, height, 3) with widht = height
    """
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
    n_samples, n_pixels = X.shape
    width = int(np.sqrt(n_pixels // 3))
    assert n_pixels == 3 * width * width
    X = X.reshape((n_samples, 3, width, width))
    X = X.transpose((0, 2, 3, 1))
    return X


def img2vec(X):
    """
    Takes images of shape (n_samples, width, height, 3) and reshape to
    (n_samples, width * height * 3)
    """
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=0)
    n_samples, width, height, n_channels = X.shape
    assert n_channels == 3
    assert width == height
    X = X.transpose((0, 3, 1, 2))
    X = X.reshape((n_samples, width * height * n_channels))
    return X


def transform_T(X, Y=None):
    """
    Takes images Xtr_reshaped of shape  (n_samples, n_pixels) and their
    classes Y (n_samples, 1), translate them 8 times,
    and returns images of shape ( 9 * n_samples, n_pixels) and their classes
    of shape (9 * n_samples, 1)
    """
    # Reshape to (n_samples, width, height, 3)
    X = vec2img(X)
    if Y is not None:
        assert len(X) == len(Y)
    n_samples = X.shape[0]
    n_rows = X.shape[1]
    n_cols = X.shape[2]

    # Direction of translation
    trans_value = 1  # How many pixels we want to translate
    x_translations = np.array([0, 1, 1, -1, -1, 0, 0, -1, 1]) * trans_value
    y_translations = np.array([0, 1, -1, -1, 1, -1, 1, 0, 0]) * trans_value

    n_translations = len(x_translations)
    tf_X = np.ones((n_samples * n_translations, n_rows, n_cols, 3))
    if Y is not None:
        tf_Y = np.ones(n_samples * n_translations)
    for i in range(n_samples):
        for j in range(n_translations):
            x_trans = x_translations[j]
            y_trans = y_translations[j]
            img = X[i]
            tf_img = tf.warp(img,
                             AffineTransform(translation=(x_trans, y_trans)))
            tf_X[i * n_translations + j] = tf_img
            if Y is not None:
                tf_Y[i * n_translations + j] = Y[i]
    # Reshape back to (n_samples, n_pixels)
    tf_X = img2vec(tf_X)
    if Y is not None:
        return tf_X, tf_Y
    else:
        return tf_X


def plot_image(img):
    if len(img.shape) == 1:
        img = vec2img(img)[0]
    assert len(img.shape) == 3
    # Convert centered values to ints between 0 and 255
    img = ((img - np.min(img))/(img.max() - img.min()) * 255).astype(np.uint8)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
