from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

from demo import Xtr, Ytr, Ytr_unique
from utils import plot_confusion_matrix

n_samples = len(Xtr)

classifier = svm.SVC(C=1., kernel='rbf', gamma=0.1)

perform_pca = False
if perform_pca:
    pca = PCA(n_components=len(Xtr[0]))
    Xtr_ = pca.fit_transform(Xtr)
    print(pca.explained_variance_ratio_)

else:
    Xtr_ = Xtr

assert len(Xtr_) == len(Ytr)
print('performing classification with {} classifier'.format(classifier))
classifier.fit(Xtr_[:n_samples / 2], Ytr[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = Ytr[n_samples / 2:]
predicted = classifier.predict(Xtr_[n_samples / 2:])

# Confusion Matrix
conf_mat = confusion_matrix(expected, predicted)
accuracy = accuracy_score(expected, predicted)
plot_confusion_matrix(conf_mat, Ytr_unique)
print('accuracy_score={}'.format(accuracy))
