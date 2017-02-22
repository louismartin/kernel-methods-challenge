import os

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Make sure the directories exist
for directory in [DATA_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

