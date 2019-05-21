#-*- coding:utf-8 -*-
import numpy as np


# 利用Python实现NCA,Neighbor Component Analysis
# 逐步求解梯度，涉及到求和利用两层for循环来解决

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
            if s % 2 == 0 and s > 1:
                print("Step {}, target = {}...".format(s, target))
            # to low dimension
            low_X = np.dot(X, self.A)

            # distance matrix
            sum_row = np.sum(low_X ** 2, axis = 1)
            xxt = np.dot(low_X, low_X.transpose())
            dist_mat = sum_row + np.reshape(sum_row, (-1, 1)) - 2 * xxt

            # distance to probability
            exp_neg_dist = np.exp(-dist_mat)
            exp_neg_dist = exp_neg_dist - np.diag(np.diag(exp_neg_dist))
            prob_mat = exp_neg_dist / np.sum(exp_neg_dist, axis = 1).reshape((-1, 1))

            # pi = \sum_{j \in C_i} p_{ij}
            prob_row = np.array([np.sum(prob_mat[i][Y == Y[i]]) for i in range(self.n_samples)])

            # target
            target = np.sum(prob_row)

            # gradients
            gradients = np.zeros((self.high_dims, self.high_dims))
            # for i
            for i in range(self.n_samples):
                k_sum = np.zeros((self.high_dims, self.high_dims))
                k_same_sum = np.zeros((self.high_dims, self.high_dims))
                # for k
                for k in range(self.n_samples):
                    out_prod = np.outer(X[i] - X[k], X[i] - X[k])
                    k_sum += prob_mat[i][k] * out_prod
                    if Y[k] == Y[i]:
                        k_same_sum += prob_mat[i][k] * out_prod
                gradients += prob_row[i] * k_sum - k_same_sum
            gradients = 2 * np.dot(gradients, self.A)

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





