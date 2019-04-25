import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline #(used in IPython)
from sklearn.datasets.samples_generator import make_classification
# 生成带标签的数据
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
X, y = make_classification(n_samples=1000, 
        n_features=3, 
        n_redundant=0, 
        n_classes=3, 
        n_informative=2, 
        n_clusters_per_class=1, 
        class_sep =0.5, 
        random_state =10)
fig = plt.figure()
# 查看数据在三维空间中的分布
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o',c=y)
plt.show()

# PCA降维(作为对比)
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
# 打印2个主成分方差比和方差
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.transform(X)
# 绘制降维后的数据分布
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
plt.show()

# LDA降维
# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X,y)
X_new = lda.transform(X)
# 绘制LDA降维
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
plt.show()