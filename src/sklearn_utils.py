import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold


def cross_validate(Xtr, Ytr, classifier, nb_folds=10):
    """
    Cross-validate and returns the predictions.
    :param _input_X: (n_samples, n_features) np.array
    :param _input_Y: (n_samples, ) np.array
    :param classifier: sklearn classifier object, with fit and predict methods
    :param nb_folds: integer
    :return: Vector of predictions
    """
    cv_folds = StratifiedKFold(Ytr, nb_folds, shuffle=True)
    pred = np.zeros(Ytr.shape)  # vector of 0 in which to store the predictions
    for tr, te in cv_folds:
        # Restrict data to train/test folds
        Xtr = np.array(Xtr)[tr, :]
        ytr = np.array(Ytr)[tr]
        Xte = np.array(Xtr)[te, :]
        yte = np.array(Ytr)[te]

        # Scale data
        scaler = preprocessing.StandardScaler()  # create scaler
        Xtr = scaler.fit_transform(Xtr)  # fit the scaler to the training data and transform training data
        Xte = scaler.transform(Xte)  # transform test data

        # Fit classifier
        classifier.fit(Xtr, ytr)

        # Predictions
        pred[te] = classifier.predict(Xte).reshape(yte.size,1)
    return pred