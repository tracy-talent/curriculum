# 降维学习

## 代码任务描述

* task1：使用PCA降维
* task2：使用SVD降维
* task3：使用ISOMAP降维

## 代码说明

* integrated.py集成了如上所述的三个任务，跑一次integrated.py将会自动化完成三种不同降维方法分别在降维维度k = 10，20，30下对两个数据集sonar和splice的进行学习和测试，生成结果将写进result/result.csv中
* 也可以单个任务的跑，PCA_centra.py，PCA_norm.py，SVD.py，ISOMAP.py分别对应其文件名所描述的任务，跑单个任务的结果不会写入到结果文件中，只会输出在控制台上
* metrics.py写的度量方式，实验要求采用的1NN计算欧氏距离，如果想采用别的度量方式，可以在metrics.py中加入度量方法

## 运行要求

* python3
* numpy