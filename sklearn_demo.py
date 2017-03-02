from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

from demo import Xtr, Ytr, Ytr_unique
from src.utils import plot_confusion_matrix

n_samples = len(Xtr)

# classifier = svm.SVC(C=1., kernel='rbf', gamma=0.1)
classifier = OneVsRestClassifier(svm.SVC(C=1., kernel='rbf', gamma=0.1))
# classifier = KNeighborsClassifier(n_neighbors=10)

# Perform PCA ?
perform_pca = True
if perform_pca:
    pca = PCA(n_components=500)
    Xtr_ = pca.fit_transform(Xtr)
    print(pca.explained_variance_ratio_)
else:
    Xtr_ = Xtr

# Fit
assert len(Xtr_) == len(Ytr)
print('performing classification with {} classifier'.format(classifier))
train_ratio = 0.8
n_train_samples = int(n_samples * train_ratio)
classifier.fit(Xtr_[:n_train_samples], Ytr[:n_train_samples])

# Predict
expected = Ytr[n_train_samples:]
print('performing prediction with {} classifier'.format(classifier))
predicted = classifier.predict(Xtr_[n_train_samples:])

# Confusion Matrix
conf_mat = confusion_matrix(expected, predicted)
accuracy = accuracy_score(expected, predicted)
plot_confusion_matrix(conf_mat,
                      Ytr_unique,
                      title="Confusion Matrix with pca={}".format(perform_pca),
                      classifier=classifier.__class__)
print('accuracy_score={}'.format(accuracy))
