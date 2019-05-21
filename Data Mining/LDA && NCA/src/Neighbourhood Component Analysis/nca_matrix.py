#-*- coding:utf-8 -*-
import numpy as np


# 利用Python实现NCA,Neighbor Component Analysis
# 利用矩阵操作取代第二层for循环，加速运算，但是会memory error，并且代码可读性变差
# 只适合小数据

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
        target = 0
        while s < self.max_steps:
            if s % 2 == 0 and s > 0:
                print("Step {}, target = {}...".format(s, target))
            # to low dimension
            low_X = np.dot(X, self.A)

            # distance matrix
            #dist_mat = np.linalg.norm(low_X[None, :, :] - low_X[:, None, :], axis = 2)
            dist_mat = np.sum((low_X[None,:,:] - low_X[:,None,:]) ** 2, axis = 2)

            # distance to probability
            exp_neg_dist = np.exp(0.0 - dist_mat)
            exp_neg_dist = exp_neg_dist - np.diag(np.diag(exp_neg_dist))
            pij_mat = exp_neg_dist / np.sum(exp_neg_dist, axis = 1)[:, None]

            # mask where mask_{ij} = True if Y[i] == Y[j], shape = (n_samples, n_samples)
            mask = (Y == Y[:, None])
            # mask array
            pij_mat_mask = pij_mat * mask
            # pi = \sum_{j \in C_i} p_{ij}
            pi_arr = np.array(np.sum(pij_mat_mask, axis = 1))

            # target
            target = np.sum(pi_arr)

            # gradients
            part_gradients = np.zeros((self.high_dims, self.high_dims))
            for i in range(self.n_samples):
                xik = X[i] - X
                prod_xik = xik[:, :, None] * xik[:, None, :]
                pij_prod_xik = pij_mat[i][:, None, None] * prod_xik
                first_part = pi_arr[i] * np.sum(pij_prod_xik, axis = 0)
                second_part = np.sum(pij_prod_xik[Y == Y[i], :, :], axis = 0)
                part_gradients += first_part - second_part
            gradients = 2 * np.dot(part_gradients, self.A)
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





