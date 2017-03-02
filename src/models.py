import cvxopt
import numpy as np

from tqdm import tqdm


class UnknownKernel(Exception):
    def __init(self, kernel):
        self.message = kernel

        
class KernelSVM:
    '''
    One-vs-all Kernel SVM implementation using a quadratic programming solver
    Methods:
        - test(X_train, Y_train)
        - predict(X_test)
     NOTE: here we assume that X_train is centered, i.e. np.mean(X_train, axis=0) == 0
    '''
    
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        
        
    def _kernel(self):
        ''' 
        Returns kernel function 
        where:
            - self.kernel: kernel type
        returns
            - kernel function (raises error UnknownKernel is kernel not known)
        '''
        if self.kernel == 'linear':
            def k(x, y):
                return np.inner(x, y)
            return k
        else:
            raise UnknownKernel(self.kernel)
    
    
    def _kernel_matrix(self, X_1, X_2):
        '''
        Computes kernel matrix 
        where:
            - X_1: [n_1 x p] matrix
            - X_2: [n_2 x p] matrix
        returns:
            - K: [n_1 x n_2] kernel matrix .t. K[i, j] = k(X_1[i, :], X_2[j, :])
        '''
        n_1 = np.shape(X_1)[0]
        n_2 = np.shape(X_2)[0]
        K = np.zeros([n_1, n_2])
        k = self._kernel()
        for i in range(n_1):
            for j in range(n_2):
                K[i, j] = k(X_1[i, :], X_2[j, :])
        return K
        
        
    def _train_one_vs_all(self, X_train, Y_train, class_id):
        '''
        Solves SVM Kernel Dual Problem 
                    min_{a \in \R^n} 1/2 a^T*K*a - a^T*y
                    s.t. \sum a_j = 0 and 0 <= a_j * y_j < = C
        where
            - K: [n_train x n_train] kernel matrix s.t  K[i, j] = <x_train_i, x_train_j>
            - Y_train: [n_train x 1] target matrix s.t. Y[i] = class(a_i) 
              Note that class_(a_i) = 0:n_classes 
            - i: class index to be set to +1 (all other classes set to -1) 
                 for 1-vs-all classification
            - C: regularization parameter, strictly positive 
        returns 
            - a[i]: weights s.t. prediction f(x) = \sum_j a_j * <x, x_j>
        '''
        # recover shape and kernel matrix
        n = Y_train.shape[0]
        K_train = self._kernel_matrix(self.X_train, self.X_train)

        # one-vs-all target 
        d = ((Y_train == class_id) * 2 -1).astype(np.float64)

        # conversion to cvxopt matrixes
        P = cvxopt.matrix(K_train)
        q = cvxopt.matrix(-d, tc='d')

        #  0 <= a_j * y_j
        G_low = np.diag(-1 * d)
        h_low = np.zeros((n, 1), dtype='float64')

        # a_j * y_j <= C
        G_high = np.diag(d)
        h_high = self.C * np.ones((n, 1), dtype='float64')

        # constraint matrixes 
        G = cvxopt.matrix(np.vstack((G_low, G_high)))
        h = cvxopt.matrix(np.vstack((h_low, h_high)))

        # solves min_a 1/2 a^T * P * a + q^T * a s.t. G*a <= h 
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h)

        self.a[class_id] = np.ravel(solution['x'])
        
        
    def train(self, X_train, Y_train):
        '''
        Performs all 1-vs-all training
        '''
        self.X_train = X_train
        self.a = {} 
        self.n_classes = len(np.unique(Y_train))
        pbar = tqdm(range(self.n_classes))
        for i in pbar:
            self._train_one_vs_all(X_train, Y_train, i)
        
        
    def _predict_one_vs_all(self, X_test, class_id):
        '''
        Predicts f(x_test_j) = \sum_i a_i * <x_test_j, x_train_i> for j = 0:n_test
        where
            - a_i: training weights for class_id
            - X_test: [n_test x p] test matrix made of n test samples x_test_j, j = 0:n_test
        returns:
            - Y_test: [n_test x 1] predicted class of the matrix
        '''
        n_train = np.shape(self.X_train)[0]
        n_test = np.shape(self.X_test)[0]
        K_test = np.zeros([n_train, n_test])
        # build kernel matrix 
        K_test = self._kernel_matrix(self.X_train, self.X_test)
        # compute Y_test
        self.Y_test[class_id, :] = np.dot(self.a[class_id], K_test) 
        
        
    def predict(self, X_test):
        self.X_test = X_test
        self.Y_test = np.zeros([self.n_classes, np.shape(self.X_test)[0]])
        pbar = tqdm(range(self.n_classes))
        for i in pbar:
            self._predict_one_vs_all(X_test, i)
        return np.argmax(self.Y_test, axis=0)