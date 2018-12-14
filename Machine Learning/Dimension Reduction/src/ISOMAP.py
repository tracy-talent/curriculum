from numpy import *
from metrics import _1NN
from queue import PriorityQueue
from os import path
import time


def loadData(filename):
    content = open(filename).readlines()
    # 分割数据和标签
    data = [list(map(float32, line.strip().split(",")[:-1])) for line in content]
    tag = [list(map(int, line.strip().split(",")[-1:])) for line in content]
    return mat(data), mat(tag)


# 计算欧氏距离
def calc_distance(dataMat):
    dataSize = len(dataMat)
    Euc_distanceMat = zeros([dataSize, dataSize], float32)
    for i in range(dataSize):
        for j in range(dataSize):
            Euc_distanceMat[i][j] = linalg.norm(dataMat[i] - dataMat[j])
    return Euc_distanceMat

# 邻接表边
class edge(object):
    def __init__(self, cost, to):
        self.cost = cost  # 边权重
        self.to = to  # 入点
    def __lt__(self, other):
        return self.cost < other.cost


# dijkstra最短路径
# @param{dist:距离矩阵, graph:邻接表表示的图, src:源点}
def dijkstra(dist, graph, src):
    que = PriorityQueue()
    que.put(edge(0, src))
    while not que.empty():
        p = que.get()
        v = p.to
        if dist[src][v] < p.cost:
            continue
        for i in range(len(graph[v])):
            if dist[src][graph[v][i].to] > dist[src][v] + graph[v][i].cost:
                dist[src][graph[v][i].to] = dist[src][v] + graph[v][i].cost
                que.put(edge(dist[src][graph[v][i].to], graph[v][i].to))


# @param{dist:距离矩阵，dims:降维的维度}
# return：降维后的矩阵
def mds(dist, dims):
    dataSize = len(dist)
    if dims > dataSize:
        print('降维的维度%d大于待降维矩阵的维度%d' % (dims, dist.shape()))
        return
    dist_i_dot_2 = zeros([dataSize], float32)
    dist_dot_j_2 = zeros([dataSize], float32)
    dist_dot_dot_2 = 0.0
    bMat = zeros([dataSize, dataSize], float32)
    for i in range(dataSize):
        for j in range(dataSize):
            dist_i_j_2 = square(dist[i][j])
            dist_i_dot_2[i] += dist_i_j_2
            dist_dot_j_2[j] += dist_i_j_2 / dataSize
            dist_dot_dot_2 += dist_i_j_2
        dist_i_dot_2[i] /= dataSize
    dist_dot_dot_2 /= square(dataSize)
    for i in range(dataSize):
        for j in range(dataSize):
            dist_i_j_2 = square(dist[i][j])
            bMat[i][j] = -0.5 * (dist_i_j_2 - dist_i_dot_2[i] - dist_dot_j_2[j] + dist_dot_dot_2)
    # 特征值和特征向量
    eigVals, eigVecs = linalg.eig(bMat)
    # 取前dims大的特征值对应的索引
    eigVals_Idx = argpartition(eigVals, -dims)[:-(dims+1):-1]
    # 构建特征值对角矩阵
    eigVals_Diag = diag(maximum(eigVals[eigVals_Idx], 0.0))
    return matmul(eigVecs[:, eigVals_Idx], sqrt(eigVals_Diag))


# param{dataMat:待降维矩阵，dims:降维的维度，KNN_K:KNN的参数K}
# return：降维后的矩阵
def isomap(dataMat, dims, KNN_K):
    set_printoptions(threshold=NaN)
    inf = float('inf')
    dataSize = len(dataMat)
    if KNN_K >= dataSize:
        raise ValueError('KNN_K的值最大为数据个数 - 1:%d' % dataSize - 1)
    Euc_distanceMat = calc_distance(dataMat)
    # 建立KNN的连接图
    knn_distanceMat = ones([dataSize, dataSize], float32) * inf
    for i in range(dataSize):
        knn_disIdx = argpartition(Euc_distanceMat[i], KNN_K)[:KNN_K + 1]
        knn_distanceMat[i][knn_disIdx] = Euc_distanceMat[i][knn_disIdx]
        for j in knn_disIdx:
            knn_distanceMat[j][i] = knn_distanceMat[i][j]
    
    # 建立邻接表
    adjacencyTable = []
    for i in range(dataSize):
        edgelist = []
        for j in range(dataSize):
            if knn_distanceMat[i][j] != inf:
                edgelist.append(edge(knn_distanceMat[i][j], j))
        adjacencyTable.append(edgelist)
    
    # 调用dijkstra求最短路
    # dist存储任意两点之间最短距离
    dist = ones([dataSize, dataSize], float32) * inf
    for i in range(dataSize):
        dist[i][i] = 0.0
        dijkstra(dist, adjacencyTable, i)
    return mds(dist, dims)


def main(resfile=None):
    DATADIR = '../dataset'
    def ftrain(name):
        return path.join(DATADIR, '{}-train.txt'.format(name))
    def ftest(name):
        return path.join(DATADIR, '{}-test.txt'.format(name))
    for fname, KNN_K in {'sonar': 6, 'splice': 4}.items():
        traindataMat, traintagMat = loadData(ftrain(fname)) 
        testdataMat, testtagMat = loadData(ftest(fname))
        for dims in [10, 20, 30]:
            timestamp = time.time()
            dataMat_dimReduced = isomap(vstack([traindataMat, testdataMat]), dims, KNN_K)
            accuracy = _1NN(dataMat_dimReduced[range(0, len(traindataMat)), :], dataMat_dimReduced[range(len(traindataMat),len(dataMat_dimReduced)), :], mat(traintagMat), mat(testtagMat))
            print("(ISOMAP on %s(%d-NN))当维度为%d时,正确率为：" % (fname, KNN_K, dims), accuracy)
            print("time used：", time.time() - timestamp, 's')
            if resfile != None:
                resfile.write('ISOMAP, %s(%d-NN), k = %d, %.16f\n' % (fname, KNN_K, dims, accuracy))


if __name__ == '__main__':
    main()