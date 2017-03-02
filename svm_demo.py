import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from numpy import random
from src.models import KernelSVM

n = 100
p = 2
X_1 = 1 * random.randn(n, p) - [1, 0]
X_2 = 0.2 * random.randn(n, p) + [1, 0]
X = np.vstack((X_1, X_2))
Y = np.hstack((np.zeros(n), np.ones(n)))

plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()

svm = KernelSVM(C=1, kernel='linear')

svm.train(X, Y)

X1_mesh, X2_mesh = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
X_new = np.vstack((np.ravel(X1_mesh), np.ravel(X2_mesh)))
Y_test = svm.predict(X_new.T)

plt.scatter(X_new.T[:, 0], X_new.T[:, 1], c = Y_test)
plt.show()