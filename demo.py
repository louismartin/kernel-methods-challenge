import os
import time

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.decomposition import PCA as PCAsklearn

from src.utils import DATA_DIR
from src.data_processing import load_images, plot_image, vec2img, img2vec, transform_T
from src.models import KernelSVM
from src.feature_extraction import Dictionary
from src.pca import PCA


Xtr_path = os.path.join(DATA_DIR, "Xtr.csv")
Ytr_path = os.path.join(DATA_DIR, "Ytr.csv")

Xtr = load_images(Xtr_path)
Ytr_csv = pd.read_csv(Ytr_path).Prediction
Ytr_unique = Ytr_csv.unique()
Ytr = np.array(Ytr_csv.tolist())

print("Loaded images - shape {}".format(Xtr.shape))

# Data augmentation
do_data_augmentation = True
if do_data_augmentation:
    Xtr_reshaped = vec2img(Xtr)
    tf_Xtr_reshaped, tf_Ytr = transform_T(Xtr_reshaped, Ytr)
    Xtr = img2vec(tf_Xtr_reshaped)
    Ytr = tf_Ytr

# Dictionary learning
do_dictionary_learning = False
if do_dictionary_learning:
    dictionary = Dictionary(n_atoms=128, atom_width=16)
    if dictionary.weights_available:
        print("Loading dictionary")
        dictionary.load()
    else:
        print("Learning dictionary")
        tic = time.time()
        dictionary.fit(Xtr)
        dictionary.save()
        print("Dictionary learned in {0:.1f}s".format(time.time() - tic))
    print("Getting dictionary representation")
    Xtr = dictionary.get_representation(Xtr)

# PCA
do_pca = True
sklearn_pca = True
if do_pca:
    tic = time.time()
    print("Applying PCA")
    if sklearn_pca:
        pca = PCAsklearn(n_components=100)
        Xtr = pca.fit_transform(Xtr)
    else:
        pca = PCA(n_components=100)
        Xtr = pca.fit(Xtr, scale=False)
    print("PCA applied in {0:.1f}s".format(time.time() - tic))

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

# Training
print("Start training")
sklearn_svm = True
tic = time.time()
if sklearn_svm:
    model = OneVsRestClassifier(svm.SVC(C=1., kernel='rbf', gamma=0.1))
    model.fit(Xtr, Ytr)
else:
    model = KernelSVM(C=1, kernel='linear')
    model.train(Xtr, Ytr)
print("Model trained in {0:.1f}s".format(time.time() - tic))

Ypred = model.predict(Xval)


accuracy = accuracy_score(Yval, Ypred)
print("Accuracy: {}".format(accuracy))
