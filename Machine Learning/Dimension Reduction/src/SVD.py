from numpy import *
from os import path
from metrics import _1NN
import time

def loadData(filename):
    content = open(filename).readlines()
    # 分割数据和标签
    data = [list(map(float32, line.strip().split(",")[:-1])) for line in content]
    tag = [list(map(int, line.strip().split(",")[-1:])) for line in content]
    return mat(data), mat(tag)

#调用svd计算出投影矩阵W
def svd(dataMat, dims):
    _, S, VT = linalg.svd(dataMat)
    # 取前dims大的奇异值对应的索引
    SIdx_topK = argpartition(S, -dims)[:-(dims+1):-1]
    # 将前dims大的奇异值对应的右奇异向量组合成投影矩阵
    projectionMat = VT[SIdx_topK, :]
    return projectionMat.T

def main(resfile=None):
    DATADIR = '../dataset'
    def ftrain(name):
        return path.join(DATADIR, '{}-train.txt'.format(name))
    def ftest(name):
        return path.join(DATADIR, '{}-test.txt'.format(name))
    for fname in ['sonar', 'splice']:
        traindataMat, traintagMat = loadData(ftrain(fname))
        testdataMat, testtagMat = loadData(ftest(fname))
        for dims in [10, 20, 30]:
            timestamp = time.time()
            projectionMat = svd(traindataMat, dims)
            # 投影
            traindataMat_proj = traindataMat * projectionMat
            testdataMat_proj = testdataMat * projectionMat
            accuracy = _1NN(traindataMat_proj, testdataMat_proj, traintagMat, testtagMat)
            print("(SVD on %s)当维度为%d时,正确率为：" % (fname, dims), accuracy) 
            print('time used:', time.time() - timestamp, 's')
            if resfile != None:
                resfile.write('SVD, %s, k = %d, %.16f\n' % (fname, dims, accuracy))

if __name__ == '__main__':
    main()