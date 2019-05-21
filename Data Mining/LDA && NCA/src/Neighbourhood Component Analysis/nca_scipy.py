#-*- coding:utf-8 -*-
import numpy as np
from scipy.optimize import minimize, fmin_cg
from sklearn.metrics import pairwise_distances
import time
import warnings
import sys
from sklearn.exceptions import ConvergenceWarning

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

# 利用Python实现NCA,Neighbor Component Analysis
# 得到loss和gradient后用scipy.optimize来求解

class NCA():
    def __init__(self, low_dims, optimizer = 'gd', max_steps = 500, tol = 1e-9, init_style = "normal", init_stddev = 0.1, verbose = True, learning_rate = None):
        '''
        init function
        @params low_dims : the dimension of transformed data
        @params optimizer : if 'gd' use gradient descent; if 'cg' use conjugate descent
        @params max_steps : the max steps of gradient descent, default 500
        @params tol : tolerance for termination
        @params init_style : parameter init_functions
                  "normal"   : init with gaussian, stddev = init_stddev
                  "uniform"  : init with uniform [0, 1]
                  "diagonal" : init only diagonal of matrix
        @params init_stddev : the sttdev of gaussian
        @params verbose : whether to print detail information
        '''
        self.low_dims = low_dims
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.tol = tol
        self.init_style = init_style
        self.init_stddev = init_stddev
        self.verbose = verbose
        self.target = 0.0                       # target value
        self.n_steps = 0                        # current step

    def fit(self, X, Y):
        '''
        train on X and Y, supervised, to learn a matrix A
        maximize \sum_i \sum_{j \in C_i} frac{exp(-||Ax_i-Ax_j||^2)}{\sum_{k neq i} exp(-||Ax_i-Ax_k||^2)}
        @params X : 2-d numpy.array
        @params Y : 1-d numpy.array
        '''
        # basic infos
        (n, d) = X.shape
        self.n_samples = n
        self.high_dims = d

        # parametric matrix
        A = self.get_random_params(shape = (self.high_dims, self.low_dims))

        if self.optimizer == 'gd':
            self.fit_gradient_descent(A, X, Y)
        elif self.optimizer == 'cd':
            self.fit_conjugate_descent(A, X, Y)
        else:
            print("No such optimizer {}, please check!".format(self.optimizer))
            raise Exception
        return self

    def fit_conjugate_descent(self, A, X, Y):
        '''
        train on X and Y, supervised, to learn a matrix A, use conjugate descent
        maximize \sum_i \sum_{j \in C_i} frac{exp(-||Ax_i-Ax_j||^2)}{\sum_{k neq i} exp(-||Ax_i-Ax_k||^2)}
        @params X : 2-d numpy.array
        @params Y : 1-d numpy.array
        '''
        start = time.time()
        def costf(A):
            f, _ = self.nca_cost(A.reshape((self.high_dims, self.low_dims)), X, Y)
            return f

        def costg(A):
            _, g = self.nca_cost(A.reshape((self.high_dims, self.low_dims)), X, Y)
            return g

        # optimizer params
        self.A = fmin_cg(costf, A.ravel(), costg, maxiter = self.max_steps)
        self.A = self.A.reshape((self.high_dims, self.low_dims))

        end = time.time()
        train_time = end - start

        # print information
        if self.verbose:
            cls_name = self.__class__.__name__
            print("[{}] Traing took {:8.2f}s.".format(cls_name, train_time))

    def fit_gradient_descent(self, A, X, Y):
        '''
        train on X and Y, supervised, to learn a matrix A, use gradient descent
        maximize \sum_i \sum_{j \in C_i} frac{exp(-||Ax_i-Ax_j||^2)}{\sum_{k neq i} exp(-||Ax_i-Ax_k||^2)}
        @params X : 2-d numpy.array
        @params Y : 1-d numpy.array
        '''
        start = time.time()
        # optimizer params
        optimizer_params = {'method' : 'L-BFGS-B',
                            'fun' : self.nca_cost,
                            'args' : (X, Y),
                            'jac' : True,
                            'x0' : A.ravel(),
                            'options' : dict(maxiter = self.max_steps),
                            'tol' : self.tol}
        opt = minimize(**optimizer_params)

        # get A
        self.A = opt.x.reshape((self.high_dims, self.low_dims))
        self.n_steps = opt.nit

        end = time.time()
        train_time = end - start

        # print information
        if self.verbose:
            cls_name = self.__class__.__name__
            if not opt.success:
                warnings.warn("[{}] NCA did not converge : {}".format(cls_name, opt.message), ConvergenceWarning)
            print("[{}] Traing took {:8.2f}s in {} steps.".format(cls_name, train_time, self.n_steps))

    def transform(self, X):
        '''
        transform X from high dimension space to low dimension space
        '''
        low_X = np.dot(X, self.A)
        return low_X

    def fit_transform(self, X, Y):
        '''
        train on X
        and then
        transform X from high dimension space to low dimension space
        '''
        self.fit(X, Y)
        low_X = self.transform(X)
        return low_X

    def nca_cost(self, A, X, Y):
        '''
        return the loss and gradients of NCA given A X Y
        @params A : 1-d numpy.array (high_dims * low_dims, )
        @params X : 2-d numpy.array (n_samples, high_dims)
        @params Y : 1-d numpy.array (n_samples, )
        '''
        # reshape A
        A = A.reshape((self.high_dims, self.low_dims))

        # to low dimension
        low_X = np.dot(X, A)

        # distance matrix-->proba_matrix
        pij_mat = pairwise_distances(low_X, squared = True)
        np.fill_diagonal(pij_mat, np.inf)
        pij_mat = np.exp(0.0 - pij_mat - logsumexp(0.0 - pij_mat, axis = 1)[:, None])

        # mask where mask_{ij} = True if Y[i] == Y[j], shape = (n_samples, n_samples)
        mask = (Y == Y[:, None])                                                        # (n_samples, n_samples)
        # mask array
        pij_mat_mask = pij_mat * mask                                                   # (n_samples, n_samples)
        # pi = \sum_{j \in C_i} p_{ij}
        pi_arr = np.array(np.sum(pij_mat_mask, axis = 1))                               # (n_samples, )
        # target
        self.target = np.sum(pi_arr)

        # gradients
        weighted_pij = pij_mat_mask - pij_mat * pi_arr[:, None]                             # (n_samples, n_samples)
        weighted_pij_sum = weighted_pij + weighted_pij.transpose()                          # (n_samples, n_samples)
        np.fill_diagonal(weighted_pij_sum, -weighted_pij.sum(axis = 0))

        gradients = 2 * (low_X.transpose().dot(weighted_pij_sum)).dot(X).transpose()        # (high_dims, low_dims)
        gradients = 0.0 - gradients

        self.n_steps += 1
        if self.verbose:
            print("Training, step = {}, target = {}...".format(self.n_steps, self.target))

        return [-self.target, gradients.ravel()]

    def get_random_params(self, shape):
        '''
        get parameter init values
        @params shape : tuple
        '''
        if self.init_style == "normal":
            return self.init_stddev * np.random.standard_normal(size = shape)
        elif self.init_style == "uniform":
            return np.random.uniform(size = shape)
        else:
            print("No such parameter init style!")
            raise Exception





