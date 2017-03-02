import numpy as np

from src.data_processing import vec2img, img2vec
# ----- Dictionary learning -----
# Inspired from https://github.com/louismartin/dictionary-learning


def initialize_atoms(Xtr, n_atoms, atom_width):
    """Initialize random atoms with high energy for dictionary learning
    returned atoms have shape (n_atoms, atom_width * atom_width * 3)
    """
    # We will work on images and atoms of shape (None, width, height, 3)
    Xtr = vec2img(Xtr)
    n_samples, img_width, _, _ = Xtr.shape

    # We will initialize more atoms than needed in order to take those with
    # the highest energies
    n_initial_atoms = 20 * n_atoms

    # Choose a random image for each atom
    img_indexes = np.random.randint(n_samples, size=n_initial_atoms)
    img_indexes = img_indexes.repeat(atom_width * atom_width)\
                             .reshape(n_initial_atoms, atom_width, atom_width)

    # Coordinates of top left corner of each atom
    max_top_left = img_width - atom_width + 1
    top_left_X = np.random.randint(max_top_left, size=n_initial_atoms)
    top_left_Y = np.random.randint(max_top_left, size=n_initial_atoms)
    top_left_X = top_left_X.repeat(atom_width * atom_width)\
                           .reshape(n_initial_atoms, atom_width, atom_width)
    top_left_Y = top_left_Y.repeat(atom_width * atom_width)\
                           .reshape(n_initial_atoms, atom_width, atom_width)

    # Atoms coordinates strictly speaking
    X, Y = np.meshgrid(range(atom_width), range(atom_width))
    X = np.expand_dims(X, axis=0).repeat(n_initial_atoms, axis=0)
    Y = np.expand_dims(Y, axis=0).repeat(n_initial_atoms, axis=0)
    X += top_left_X
    Y += top_left_Y

    atoms = Xtr[img_indexes, Y, X, :]

    # Select atoms with highest energy
    energies = np.sum(atoms.reshape((n_initial_atoms, -1))**2, axis=1)
    indexes = np.argsort(energies)[::-1][:n_atoms]
    atoms = atoms[indexes]

    # Cast back from (n_atoms, width, height, 3) to (n_atoms, n_pixels)
    atoms = img2vec(atoms)
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
    for i in range(iterations):
        error = np.dot(coefs, atoms) - data
        coefs = sparsity_projection(coefs - lambd * np.dot(error, atoms.T),
                                    sparsity)
    return coefs
