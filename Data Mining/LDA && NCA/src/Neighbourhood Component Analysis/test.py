#-*- coding:utf-8 -*-


from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

#from nca_naive import NCA
#from nca_matrix import NCA
#from nca_fast import NCA
from nca_scipy import NCA



mnist = input_data.read_data_sets("/home/lxcnju/workspace/datasets/mnist/")

num = 1000

pca = PCA(n_components = 100)

X = mnist.train.images[0 : num]
Y = mnist.train.labels[0 : num]
uni_Y = np.unique(Y)


X = pca.fit_transform(X)

print("Beigin...")
print(X.shape)
print(Y.shape)
print(uni_Y)


nca = NCA(low_dims = 10, optimizer = 'cd', learning_rate = 0.01)

nca.fit(X, Y)

low_x = nca.transform(X)

print("Done!")

