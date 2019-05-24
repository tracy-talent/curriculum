import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.preprocessing import label_binarize
 
if __name__ == '__main__':
    np.random.seed(0)
    iris = load_iris()
    # data = pd.read_csv('iris.data', header = None)  #读取数据
    iris_types = np.unique(iris.target)
    n_class = iris_types.size
    x = iris.data[:, :2]  #只取前面两个特征
    y = pd.Categorical(iris.target).codes  #将标签转换0,1,...
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.6, random_state = 0)
    y_one_hot = label_binarize(y_test, np.arange(n_class))  #装换成类似二进制的编码
    alpha = np.logspace(-2, 2, 20)  #设置超参数范围
    model = LogisticRegressionCV(Cs = alpha, cv = 3, penalty = 'l2')  #使用L2正则化
    model.fit(x_train, y_train)
    print('超参数：', model.C_)
    # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
    y_score = model.predict_proba(x_test)
    # 1、调用函数计算micro类型的AUC
    print('调用函数auc：', metrics.roc_auc_score(y_one_hot, y_score, average='micro'))
    # 2、手动计算micro类型的AUC
    #首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
    fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(),y_score.ravel())
    print(y_score.shape)
    print(thresholds, thresholds.shape)
    print(fpr, fpr.shape)
    auc = metrics.auc(fpr, tpr)
    print('手动计算auc：', auc)
    #绘图
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    #FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'鸢尾花数据Logistic分类后的ROC和AUC', fontsize=17)
    plt.show()