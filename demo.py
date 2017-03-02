import os

import pandas as pd
import numpy as np

from src.utils import DATA_DIR
from src.data_processing import load_images, plot_image, vec2img, img2vec, transform_T


Xtr_path = os.path.join(DATA_DIR, "Xtr.csv")
Ytr_path = os.path.join(DATA_DIR, "Ytr.csv")

Xtr = load_images(Xtr_path)
Ytr_csv = pd.read_csv(Ytr_path).Prediction
Ytr_unique = Ytr_csv.unique()
Ytr = np.array(Ytr_csv.tolist())

print("Loaded images of shape {}".format(Xtr.shape))

plot_image(Xtr[1])

data_augmente = True
if data_augmente:

    Xtr_reshaped = vec2img(Xtr)
    tf_Xtr_reshaped, tf_Ytr = transform_T(Xtr_reshaped, Ytr)
    Xtr = img2vec(tf_Xtr_reshaped)