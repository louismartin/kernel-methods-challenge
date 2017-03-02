import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold


def cross_validate(_input_X, _input_Y, classifier, nb_folds=10):
    """
    Cross-validate and returns the predictions.
    :param _input_X: (n_samples, n_features) np.array
    :param _input_Y: (n_samples, ) np.array
    :param classifier: sklearn classifier object, with fit and predict methods
    :param nb_folds: integer
    :return: Vector of predictions
    """
    SKF = KFold(n_splits=nb_folds, random_state=7)
    cv_folds = SKF.split(np.array(_input_X), np.array(_input_Y).reshape(_input_Y.shape[0]))
    pred = np.zeros(_input_Y.shape)

    for tr, te in cv_folds:

        Xtr = np.array(_input_X)[tr, :]
        ytr = np.array(_input_Y)[tr]
        Xte = np.array(_input_X)[te, :]
        yte = np.array(_input_Y)[te]

        # Scale
        scaler = preprocessing.StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        # Fit
        classifier.fit(Xtr, ytr)

        # Predictions
        pred[te] = classifier.predict(Xte).reshape(yte.size,1)
    return pred