from collections import Counter
import os
import time

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.decomposition import PCA as PCAsklearn

from src.utils import DATA_DIR
from src.data_processing import load_images, transform_T
from src.models import KernelSVM
from src.feature_extraction import Dictionary
from src.pca import PCA


def learn(X, Y):
    pca = None
    dictionary = None
    model = None

    # Data augmentation
    if DO_DATA_AUGMENTATION:
        print("Augmenting data")
        X, Y = transform_T(X, Y)
        print("Number of samples augmented to {}".format(X.shape[0]))

    # Dictionary learning
    if DO_DICTIONARY_LEARNING:
        dictionary = Dictionary(n_atoms=128, atom_width=16)
        if dictionary.weights_available:
            print("Loading dictionary")
            dictionary.load()
        else:
            print("Learning dictionary")
            tic = time.time()
            dictionary.fit(X)
            dictionary.save()
            print("Dictionary learned in {0:.1f}s".format(time.time() - tic))
        print("Getting dictionary representation")
        X = dictionary.get_representation(X)

    # PCA
    sklearn_pca = True
    if DO_PCA:
        tic = time.time()
        print("Applying PCA")
        n_components = 100
        if sklearn_pca:
            pca = PCAsklearn(n_components=n_components)
            X = pca.fit_transform(X)
            print("Variance explained: {:.2f}".format(
                      np.sum(pca.explained_variance_ratio_)))
        else:
            pca = PCA(n_components=n_components)
            X = pca.fit(X, scale=False)
            print("Variance explained: {:.2f}".format(
                      np.sum(pca.e_values_ratio_)))
        print("PCA applied in {0:.1f}s".format(time.time() - tic))

    # Training
    print("Starting training")
    sklearn_svm = True
    tic = time.time()
    if sklearn_svm:
        model = OneVsRestClassifier(svm.SVC(C=1., kernel='rbf', gamma=0.1))
        model.fit(X, Y)
    else:
        model = KernelSVM(C=1, kernel='linear')
        model.train(X, Y)
    print("Model trained in {0:.1f}s".format(time.time() - tic))

    return pca, dictionary, model


def transform(X):
    # Data augmentation
    if DO_DATA_AUGMENTATION:
        print("Augmenting data")
        X = transform_T(X)
        print("Number of samples augmented to {}".format(X.shape[0]))

    # Dictionary learning
    if DO_DICTIONARY_LEARNING:
        print("Getting dictionary representation")
        X = dictionary.get_representation(X)

    # PCA
    if DO_PCA:
        tic = time.time()
        print("Applying PCA")
        sklearn_pca = True
        if sklearn_pca:
            X = pca.transform(X)
        else:
            X = pca.transform(X, scale=False)
        print("PCA applied in {0:.1f}s".format(time.time() - tic))
    return X


def predict(X):
    Ypred = model.predict(X)

    if DO_DATA_AUGMENTATION:
        # Take the class with the most votes for all translations
        n_val_samples = len(X) // 9
        Yvote = np.zeros(n_val_samples)
        for i in range(n_val_samples):
            pred = int(Counter(Ypred[i*9:(i+1)*9]).most_common(1)[0][0])
            Yvote[i] = pred
        Ypred = Yvote
    return Ypred


def write_submission(Yte, Yte_path):
    assert len(Yte) == 2000
    df = pd.DataFrame(index=np.arange(1, len(Yte) + 1),
                      data=Yte.astype(int),
                      columns=["Prediction"])
    df.to_csv(Yte_path, index_label="Id")


# Global variables
DO_DATA_AUGMENTATION = False
DO_DICTIONARY_LEARNING = False
DO_PCA = True

Xtr_path = os.path.join(DATA_DIR, "Xtr.csv")
Xte_path = os.path.join(DATA_DIR, "Xte.csv")
Ytr_path = os.path.join(DATA_DIR, "Ytr.csv")
Yte_path = os.path.join(DATA_DIR, "Yte.csv")

Xtr = load_images(Xtr_path)
Xte = load_images(Xte_path)
Ytr_csv = pd.read_csv(Ytr_path).Prediction
Ytr_unique = Ytr_csv.unique()
Ytr = np.array(Ytr_csv.tolist())

print("Loaded images - shape {}".format(Xtr.shape))


# Train / validation split
train_ratio = 0.8
n_samples = Xtr.shape[0]
n_train_samples = int(n_samples * train_ratio)
Xval = Xtr[n_train_samples:]
Yval = Ytr[n_train_samples:]
Xtr = Xtr[:n_train_samples]
Ytr = Ytr[:n_train_samples]
print("Training on {} samples, validating on {} samples".format(
          Xtr.shape[0], Xval.shape[0]))
print("Features: {}".format(Xtr.shape[1]))

pca, dictionary, model = learn(Xtr, Ytr)

Xval = transform(Xval)

Ypred = predict(Xval)

accuracy = accuracy_score(Yval, Ypred)
print("Accuracy: {}".format(accuracy))

Xte = transform(Xte)

Yte = predict(Xte)

write_submission(Yte, Yte_path)
