#-*- coding:utf-8 -*-


from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from nca_scipy import NCA

markers = ".,o^<>8*h+dxXH0123456789"


def test_on_mnist(n_classes = 3):
    # get data from mnist
    mnist = input_data.read_data_sets("/home/lxcnju/workspace/datasets/mnist/")
    num = int(10000 / n_classes)
    raw_X = mnist.train.images[0 : num]
    raw_Y = mnist.train.labels[0 : num]
    raw_uni_Y = np.unique(raw_Y)

    # select only several classes
    X = []
    Y = []
    uni_Y = []
    for c in raw_uni_Y[0 : n_classes]:
        X.extend(list(raw_X[raw_Y == c]))
        Y.extend(list(raw_Y[raw_Y == c]))
        uni_Y.append(c)
    X = np.array(X)
    Y = np.array(Y)
    uni_Y = np.array(uni_Y)
    classes_name = list(uni_Y)

    # first use PCA to reduce dimension
    print("PCA on mnist...")
    pca = PCA(n_components = 100)
    X = pca.fit_transform(X)

    print("Beigin on mnist...")
    print(X.shape)
    print(Y.shape)
    print(uni_Y)

    # NCA
    print("NCA on mnist...")
    nca = NCA(low_dims = 2, optimizer = 'gd', max_steps = 500, verbose = True)
    nca.fit(X, Y)
    low_x = nca.transform(X)

    # draw pics
    plt.figure()
    for i in uni_Y:
        cl_X = X[Y == i]
        plt.scatter(cl_X[:, 0], cl_X[:, 1], marker = markers[i])

    plt.legend(classes_name)
    plt.savefig("mnist_with_{}_digits.jpg".format(n_classes))
    plt.show()

    print("Done on mnist!")


# test on mnist
for n_classes in range(2, 11):
    test_on_mnist(n_classes = n_classes)

