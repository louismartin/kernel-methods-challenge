import numpy as np
from tqdm import tqdm

from src.data_processing import vec2img, img2vec
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
    atoms = random_patches(Xtr, 20 * n_atoms, atom_width)
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
    norms = np.tile(norms, (3072, 1)).T
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
