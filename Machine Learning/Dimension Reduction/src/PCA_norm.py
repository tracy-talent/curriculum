'''
中心化,标准化训练、测试数据
'''
from numpy import *
from os import path
from metrics import _1NN
import time

# 中心化,标准化数据
def norm(dataMat):
    meanVals = mean(dataMat, axis=0)
    stdVals = std(dataMat, axis=0)
    normMat = (dataMat - meanVals) / stdVals
    return normMat

def loadData(filename):
    content = open(filename).readlines()
    # 分割数据和标签
    data = [list(map(float32, line.strip().split(",")[:-1])) for line in content]
    tag = [list(map(int, line.strip().split(",")[-1:])) for line in content]
    return mat(data), mat(tag)


def pca(normMat, dims):
    # 计算协方差矩阵，除数n-1是为了得到协方差的无偏估计
    # cov(X,0) = cov(X) 除数是n-1(n为样本个数)
    # cov(X,1) 除数是n
    covMat = cov(normMat, rowvar=0)
    # 计算协方差矩阵的特征值及对应的特征向量
    eigVals, eigVects = linalg.eig(covMat)
    # 取前dims大的特征值对应的索引
    eigVals_Idx = argpartition(eigVals, -dims)[:-(dims+1):-1]
    # 将前dims大的特征值对应的特征向量组合成投影矩阵
    projectionMat = eigVects[:, eigVals_Idx]
    return projectionMat

def main(resfile=None):
    DATADIR = '../dataset'
    def ftrain(name):
        return path.join(DATADIR, '{}-train.txt'.format(name))
    def ftest(name):
        return path.join(DATADIR, '{}-test.txt'.format(name))
    for fname in ['sonar', 'splice']:
        traindataMat, traintagMat = loadData(ftrain(fname))
        testdataMat, testtagMat = loadData(ftest(fname))
        traindataMat = norm(traindataMat)
        testdataMat = norm(testdataMat)
        for dims in [10, 20, 30]:
            timestamp = time.time()
            projectionMat = pca(traindataMat, dims)
            # 投影
            traindataMat_proj = traindataMat * projectionMat
            testdataMat_proj = testdataMat * projectionMat
            accuracy = _1NN(traindataMat_proj, testdataMat_proj, traintagMat, testtagMat)
            print("(PCA on %s)当维度为%d时,正确率为：" % (fname, dims), accuracy) 
            print('time used:', time.time() - timestamp, 's')
            if resfile != None:
                resfile.write('PCA(norm), %s, k = %d, %.16f\n' % (fname, dims, accuracy))

if __name__ == '__main__':
    main()


