import os

from src.utils import DATA_DIR
from src.data_processing import load_images, plot_image


Xtr_path = os.path.join(DATA_DIR, "Xtr.csv")
Xtr = load_images(Xtr_path)

print("Loaded images of shape {}".format(Xtr.shape))

plot_image(Xtr[1])
