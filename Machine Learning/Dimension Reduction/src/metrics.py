from numpy import *

#1NN算法
def _1NN(traindataMat, testdataMat, traintagMat, testtagMat):
	trainSize = traindataMat.shape[0]
	testSize = testdataMat.shape[0]
	cnt_correct = 0
	
	for i in range(testSize):
		sample_test = mat(testdataMat[i, :])
		pos = -1
		for j in range(trainSize):
			sample_train = mat(traindataMat[j, :])
			if j == 0:
				min_dis = linalg.norm(sample_train - sample_test)
				pos = 0
			else:
				dis = linalg.norm(sample_train - sample_test)
				if dis < min_dis:
					min_dis = dis
					pos = j
	    #如果命中则cnt_correct+1
		if(testtagMat[i] == traintagMat[pos]):
			cnt_correct = cnt_correct + 1
	return cnt_correct/testSize