# Run Classifiers

## 实验环境

> python 3.6.7
>
> configparser 3.7.4
>
> scikit-learn 0.20.2
>
> numpy 1.15.4
>
> matplotlib 3.0.3

docs目录下放的报告，lab目录下放的实验相关内容

lab目录下内容:

* input是输入数据集文件目录
* output是分类评测结果存放目录
* imgs是图片输出目录
* src是存放分类模型.py代码文件的目录
* config.ini是配置文件
* paint存放绘图.py代码文件的目录

所有的.py代码文件直接在命令行下进入到代码所在目录，通过python *.py执行，不带命令行参数，运行结果以markdown表格形式存放在output目录下。执行paint目录下.py文件则是通过读取output目录中的结果来绘图。

下面对src目录下各个.py代码功能作一个简介

* preprocess.py

  ```
  在其它.py文件中导入，用于加载.arff数据集以及数据清洗
  ```

* decision_tree.py

  ```
  决策树分类器在10个数据集上的acc/auc，结果写在decision_tree_gini.md中
  ```

* knn.py

  ```
  K近邻在10个数据集上的acc/auc，结果写在knn.md中
  ```

* mlp.py

  ```
  多层感知机分类器在10个数据集上的acc/auc，结果写在mlp_100h.md中
  ```

* naive_bayes.py

  ```
  高斯朴素贝叶斯分类器分类器在10个数据集上的acc/auc，结果存放在naive_bayes_gaussian.md中
  ```

* svm.py

  ```
  支持向量机分类器在10个数据集上的acc/auc，结果存放在linearsvm.md中
  ```

* bagging_acc.py

  ```
  bagging with other classifiers在10个数据集上的acc，结果存放在bagging_acc.md中
  ```

* bagging_auc.py

  ```
  bagging with other classifiers在10个数据集上的auc，结果存放在bagging_auc.md中
  ```

* baggingknn_norm_diff_acc.py

  ```
  bagging with knn在10个数据集的原数据和标准化数据上的acc，结果存放在bagging_knn_isnorm_acc.md中
  ```

* baggingknn_norm_diff_auc.py

  ```
  bagging with knn在10个数据集的原数据和标准化数据上的auc，结果存放在bagging_knn_isnorm_auc.md中
  ```

* waveform-5000-lda-acc.py

  ```
  各分类模型在waveform-5000数据集上的原数据和LDA降维数据上的acc，结果存放在waveform-5000-lda-acc.md中
  ```

* waveform-5000-lda-auc.py

  ```
  各分类模型在waveform-5000数据集上的原数据和LDA降维数据上的auc，结果存放在waveform-5000-lda-auc.md中
  ```
