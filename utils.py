import os

import itertools
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.realpath(__file__))
CM_DIR = os.path.join(PATH, "conf_mat")


def plot_confusion_matrix(cm, classes,
                          normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Reds, directory=CM_DIR, classifier="no classifier specified"):
    """
    This function prints and plots the confusion matrix.
    It also saves it in a directory called conf_mat
    Normalization can be applied by setting `normalize=True`.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    saving_dir = os.path.join(directory, title + "_{}".format(classifier) + ".png")
    plt.savefig(saving_dir)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')