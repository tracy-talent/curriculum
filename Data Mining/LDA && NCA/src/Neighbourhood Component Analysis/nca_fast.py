#-*- coding:utf-8 -*-
import numpy as np


# 利用Python实现NCA,Neighbor Component Analysis
# 另外一种求梯度的方法，矩阵操作，避免for循环
# 适合大数据，速度和空间都很优秀

class NCA():
    def __init__(self, low_dims, learning_rate = 0.01, max_steps = 500, init_style = "normal", init_stddev = 0.1):
        '''
        init function
        @params low_dims : the dimension of transformed data
        @params learning_rate : default 0.01
        @params max_steps : the max steps of gradient descent, default 500
        '''
        self.low_dims = low_dims
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.init_style = init_style
        self.init_stddev = init_stddev
        self.target = 0.0

    def fit(self, X, Y):
        '''
        train on X and Y, supervised, to learn a matrix A
        maximize \sum_i \sum_{j \in C_i} frac{exp(-||Ax_i-Ax_j||^2)}{\sum_{k neq i} exp(-||Ax_i-Ax_k||^2)}
        @params X : 2-d numpy.array
        @params Y : 1-d numpy.array
        '''
        (n, d) = X.shape
        self.n_samples = n
        self.high_dims = d

        # parametric matrix
        self.A = self.get_random_params(shape = (self.high_dims, self.low_dims))

        # training using gradient descent
        s = 0
        while s < self.max_steps:
            if s > 0 and s % 2 == 0:
                print("Train step {}, target = {}...".format(s, self.target))
            # to low dimension
            low_X = np.dot(X, self.A)

            # distance matrix-->proba_matrix
            sum_row = np.sum(low_X ** 2, axis = 1)
            xxt = np.dot(low_X, low_X.transpose())
            pij_mat = sum_row + np.reshape(sum_row, (-1, 1)) - 2 * xxt                      # (n_samples, n_samples)
            pij_mat = np.exp(0.0 - pij_mat)
            np.fill_diagonal(pij_mat, 0.0)
            pij_mat = pij_mat / np.sum(pij_mat, axis = 1)[:, None]                          # (n_samples, n_samples)

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
            # update
            self.A += self.learning_rate * gradients

            # step++
            s += 1

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





