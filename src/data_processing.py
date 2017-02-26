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


def img2vec(X_reshaped):
    """
    Takes images of shape (n_samples, 32, 32 ,3) and reshape to (n_samples, 3072)

    """
    assert X_reshaped.shape[1:] == (32, 32, 3)
    n_samples = X_reshaped.shape[0]
    X_reshaped = X_reshaped.transpose((0, 3, 1, 2))
    X_reshaped = X_reshaped.reshape((n_samples, 3072,))

    return X_reshaped


def transform_T(Xtr_reshaped, Ytr):
    """
    Takes images Xtr_reshaped of shape  (n_samples, 32, 32 ,3) and their classes Ytr (n_samples, 1),
    translate them 8 times,
    and returns images of shape ( 9 * n_samples, 32, 32 ,3) and their classes of shape (9 * n_samples, 1)
    """
    size_sample = len(Xtr_reshaped)
    number_rows = len(Xtr_reshaped[0])
    number_cols = len(Xtr_reshaped[0][0])

    # Direction of translation
    x_translations = [0, 1, 1]#, -1, -1, 0, 0, -1, 1]
    y_translations = [0, 1, -1]#, -1, 1, -1, 1, 0, 0]

    tf_Xtr_reshaped = np.ones((size_sample * len(x_translations), number_rows, number_cols, 3),)
    tf_Ytr = np.ones(size_sample * len(x_translations),)
    for j in xrange(len(x_translations)):
        x_trans = x_translations[j]
        y_trans = y_translations[j]
        for i in xrange(size_sample):
            tf_Xtr_reshaped[i - 1 + j * size_sample] = tf.warp(Xtr_reshaped[i - 1],
                                                               AffineTransform(translation=(x_trans, y_trans)))
            tf_Ytr[i - 1 + j * size_sample] = Ytr[i - 1]
    return tf_Xtr_reshaped, tf_Ytr


def plot_image(img):
    assert img.shape == (3072,)
    img = vec2img(img)[0]
    # Convert centered values to ints between 0 and 255
    img = ((img - np.min(img))/(img.max() - img.min()) * 255).astype(np.uint8)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
