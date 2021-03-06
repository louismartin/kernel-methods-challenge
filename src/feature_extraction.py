import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data_processing import vec2img, img2vec
from src.utils import DATA_DIR
# ----- Dictionary learning -----
# Inspired from https://github.com/louismartin/dictionary-learning


def random_patches(Xtr, n_patches, patch_width):
    """Sample patches from images in Xtr"""
    # We will work on images and patches of shape (None, width, height, 3)
    Xtr = vec2img(Xtr)
    n_samples, img_width, _, _ = Xtr.shape

    # Choose a random image for each atom
    img_indexes = np.random.randint(n_samples, size=n_patches)
    img_indexes = img_indexes.repeat(patch_width * patch_width)\
                             .reshape(n_patches, patch_width, patch_width)

    # Coordinates of top left corner of each atom
    max_top_left = img_width - patch_width + 1
    top_left_X = np.random.randint(max_top_left, size=n_patches)
    top_left_Y = np.random.randint(max_top_left, size=n_patches)
    top_left_X = top_left_X.repeat(patch_width * patch_width)\
                           .reshape(n_patches, patch_width, patch_width)
    top_left_Y = top_left_Y.repeat(patch_width * patch_width)\
                           .reshape(n_patches, patch_width, patch_width)

    # patches coordinates strictly speaking
    X, Y = np.meshgrid(range(patch_width), range(patch_width))
    X = np.expand_dims(X, axis=0).repeat(n_patches, axis=0)
    Y = np.expand_dims(Y, axis=0).repeat(n_patches, axis=0)
    X += top_left_X
    Y += top_left_Y

    patches = Xtr[img_indexes, Y, X, :]
    # Cast back from (n_patches, width, height, 3) to (n_patches, n_pixels)
    patches = img2vec(patches)
    return patches


def initialize_atoms(Xtr, n_atoms, atom_width):
    """Initialize random atoms with high energy for dictionary learning
    returned atoms have shape (n_atoms, atom_width * atom_width * 3)
    """
    # We will initialize more atoms than needed in order to take those with
    # the highest energies
    atoms = random_patches(Xtr, 100 * n_atoms, atom_width)
    # Select atoms with highest energy
    energies = np.sum(atoms**2, axis=1)
    indexes = np.argsort(energies)[::-1][:n_atoms]
    atoms = atoms[indexes]
    return atoms


def sparsity_projection(coefs, sparsity):
    """Keep only k largest coefs on each row"""
    # Find the kth largest elements (pivots)
    pivots = np.partition(abs(coefs), -sparsity, axis=1)[:, -sparsity]
    to_keep = abs(coefs) >= pivots[np.newaxis].T
    # Only keep these kth largest coefficients
    # (careful sometimes more than k elements kept due to equal values)
    coefs = coefs * to_keep
    return coefs


def sparse_coding(data, atoms, coefs, sparsity, iterations=100):
    """Get the sparse representation of data on atoms represented by coefs
    Shapes:
        data: (n_samples, n_pixels)
        atoms: (n_atoms, n_pixels)
        coefs: (n_samples, n_atoms)
    """
    lambd = 1/np.linalg.norm(np.dot(atoms.T, atoms))
    for i in tqdm(range(iterations)):
        error = np.dot(coefs, atoms) - data
        coefs = sparsity_projection(coefs - lambd * np.dot(error, atoms.T),
                                    sparsity)
    return coefs


def dictionary_projection(atoms):
    """Scale all atoms to unit norm"""
    norms = np.linalg.norm(atoms, axis=1)
    norms = np.tile(norms, (atoms.shape[1], 1)).T
    atoms = atoms / norms
    return atoms


def dictionary_update(data, atoms, coefs, iterations=100):
    """Update atoms given fixed coefs, to represent the data
    Shapes:
        data: (n_samples, n_pixels)
        atoms: (n_atoms, n_pixels)
        coefs: (n_samples, n_atoms)
    """
    lambd = 1/np.linalg.norm(np.dot(coefs.T, coefs))
    for i in tqdm(range(iterations)):
        error = np.dot(coefs, atoms) - data
        atoms = dictionary_projection(atoms - lambd * np.dot(coefs.T, error))
    return atoms


def learn_dictionary(Xtr, n_atoms, atom_width, plot=False):
    n_samples = Xtr.shape[0]
    atoms = initialize_atoms(Xtr, n_atoms=n_atoms, atom_width=atom_width)
    n_patches = 10 * n_samples
    data = random_patches(Xtr, n_patches, atom_width)
    coefs = np.random.rand(n_patches, n_atoms)

    # Learn the dictionary
    iterations = 100
    errors = np.zeros(2*iterations)
    for i in tqdm(range(iterations)):
        # Sparse coding
        coefs = sparse_coding(data, atoms, coefs, sparsity=5, iterations=50)
        errors[2*i] = np.linalg.norm(np.dot(coefs, atoms) - data)**2

        # Dictionary update
        atoms = dictionary_update(data, atoms, coefs, iterations=50)
        errors[2*i+1] = np.linalg.norm(np.dot(coefs, atoms) - data)**2
    path = os.path.join(DATA_DIR, "dictionary_learning_errors.npy")
    np.save(path, errors)
    if plot:
        plt.plot(errors)
    return atoms


def extract_all_patches(Xtr, patch_width):
    """Extract all non overlapping patches of the images in Xtr"""
    # We will work on images and patches of shape (None, width, height, 3)
    Xtr = vec2img(Xtr)
    img_width = Xtr.shape[1]
    patch_per_side = img_width // patch_width
    n_patches = patch_per_side**2

    coords = np.arange(0, img_width - patch_width + 1, patch_width)
    top_left_X, top_left_Y = np.meshgrid(coords, coords)
    top_left_X = top_left_X.flatten()
    top_left_Y = top_left_Y.flatten()
    top_left_X = top_left_X.repeat(patch_width * patch_width)\
                           .reshape(n_patches, patch_width, patch_width)
    top_left_Y = top_left_Y.repeat(patch_width * patch_width)\
                           .reshape(n_patches, patch_width, patch_width)
    X, Y = np.meshgrid(range(patch_width), range(patch_width))
    X = np.expand_dims(X, axis=0).repeat(n_patches, axis=0)
    Y = np.expand_dims(Y, axis=0).repeat(n_patches, axis=0)
    X += top_left_X
    Y += top_left_Y

    patches = Xtr[:, Y, X, :]
    return patches


class Dictionary:
    def __init__(self, n_atoms, atom_width):
        self.n_atoms = n_atoms
        self.atom_width = atom_width
        filename = "atoms_{}_{}.npy".format(self.atom_width, self.n_atoms)
        self.save_path = os.path.join(DATA_DIR, filename)

    @property
    def weights_available(self):
        return os.path.exists(self.save_path)

    def fit(self, Xtr):
        self.atoms = learn_dictionary(Xtr,
                                      n_atoms=self.n_atoms,
                                      atom_width=self.atom_width,
                                      plot=False)

    def get_representation(self, Xtr, iterations=200):
        n_samples = Xtr.shape[0]

        # Extract all non overlapping patches of each image
        patches = extract_all_patches(Xtr, patch_width=self.atom_width)
        n_patches = patches.shape[1]
        patches = patches.reshape(n_samples * n_patches,
                                  self.atom_width, self.atom_width, 3)
        patches = img2vec(patches)

        # Get sparse representation of all patches
        coefs = np.random.rand(patches.shape[0], self.n_atoms)
        coefs = sparse_coding(patches, self.atoms, coefs,
                              sparsity=self.n_atoms//2, iterations=iterations)
        coefs = coefs.reshape(n_samples, self.n_atoms * n_patches)
        return coefs

    def save(self):
        np.save(self.save_path, self.atoms)

    def load(self):
        self.atoms = np.load(self.save_path)
