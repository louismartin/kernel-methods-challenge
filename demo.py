import os

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from src.utils import DATA_DIR
from src.data_processing import load_images, plot_image, vec2img, img2vec, transform_T
from src.models import KernelSVM


Xtr_path = os.path.join(DATA_DIR, "Xtr.csv")
Ytr_path = os.path.join(DATA_DIR, "Ytr.csv")

Xtr = load_images(Xtr_path)
Ytr_csv = pd.read_csv(Ytr_path).Prediction
Ytr_unique = Ytr_csv.unique()
Ytr = np.array(Ytr_csv.tolist())

print("Loaded images - shape {}".format(Xtr.shape))


data_augment = True
if data_augment:

    Xtr_reshaped = vec2img(Xtr)
    tf_Xtr_reshaped, tf_Ytr = transform_T(Xtr_reshaped, Ytr)
    Xtr = img2vec(tf_Xtr_reshaped)
    Ytr = tf_Ytr

# Train / validation split
train_ratio = 0.8
n_samples = Xtr.shape[0]
n_train_samples = int(n_samples * train_ratio)
Xval = Xtr[n_train_samples:]
Yval = Ytr[n_train_samples:]
Xtr = Xtr[:n_train_samples]
Ytr = Ytr[:n_train_samples]

svm = KernelSVM(C=1, kernel='linear')
svm.train(Xtr, Ytr)

Ypred = svm.predict(Xval)

accuracy = accuracy_score(Yval, Ypred)
print(accuracy)
