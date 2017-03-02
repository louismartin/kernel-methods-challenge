import numpy as np


class pca_:

    def __init__(self, n_components):

        self.n_components = n_components
        self.explained_variance_ratio_ = None
        self.explained_variance_ = None

    def fit_transform(self, X, scale=True):

        n_samples, n_features = X.shape

        if scale:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        print('computing pca with SVD')
        u, s, v = np.linalg.svd(X)

        _e_vectors = u[:, :self.n_components]
        _e_vectors *= s[:self.n_components]

        explained_variance_ = (s ** 2) / n_samples
        self.explained_variance_ = explained_variance_[:self.n_components]

        total_var = float(explained_variance_.sum())
        self.explained_variance_ratio_ = [float(e) / total_var for e in self.explained_variance_]

        return _e_vectors
