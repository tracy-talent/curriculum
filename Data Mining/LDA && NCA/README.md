
## 一、Linear Discriminant Analysis(LDA)

### 1.1 Rationale

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;线性判别分析(LDA)是一种监督学习的分类和降维的方法，但更多是被用来降维。LDA的原理是让投影后同一类中数据的投影点之间尽可能地靠近，而类不同类别中数据的类别中心之间的距离尽可能远，用一句话概括就是“投影后类内方差最小，类间方差最大”。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;假设我们的数据集D={(x1,y1),(x2,y2),...,((xm,ym))}，其中任意样本xixi为n维向量，yi$$\in$${C1,C2,...,Ck}。我们定义Nj(j=1,2...k)为第j类样本的个数，Xj(j=1,2...k)为第j类样本的集合，而μj(j=1,2...k)为第j类样本的均值向量，定义Σj(j=1,2...k)为第j类样本的协方差矩阵。由于我们是多类向低维投影，则此时投影到的低维空间就不是一条直线，而是一个超平面了。假设我们投影到的低维空间的维度为d，对应的基向量为(w1,w2,...wd)，基向量组成的矩阵为W, 它是一个n×d的矩阵。此时优化的目标变成
$$
\frac{W^TS_bW}{W^TS_wW}
$$
其中$$S_b=\sum_{j=1}^{k}N_{j}(\mu_{j}-\mu)(\mu_j-\mu)^T$$，$$\mu$$为所有样本均值向量。$$S_{\mathcal{w}}=\sum_{j=1}^{k}\sum_{x \in X_{j}}(x-\mu_j)(x-\mu_j)^T$$。但是有一个问题，就是$$W^TS_bW$$和$$W^TS_{\mathcal{w}}W$$都是矩阵，不是标量，无法作为一个标量函数来优化。可以用如下的替代优化目标
$$
\begin{aligned}
\mathop{\arg\max_{\mathrm{W}}}\ \| \mathrm{J}(\mathrm{W})\|&=\frac{\prod_{diag}W^TS_bW}{\prod_{diag}W^TS_{\mathcal{w}}W}\\
& =\prod_{i=1}^{d}\frac{\mathcal{w}_i^TS_b\mathcal{w}_i}{\mathcal{w}_i^TS_\mathcal{w}\mathcal{w}_i}
\end{aligned}
$$
其中$$\prod_{diag}A$$为A的主对角线元素的乘积。利用广义瑞利商可知，最大值是$$S_{w}^{-1}S_b$$的最大特征值，而最大的d个值的乘积就是$$S_{w}^{-1}S_b$$最大的d个特征值的乘积，而W即为$$S_{w}^{-1}S_b$$最大的d个特征值对应的d个特征向量张成的矩阵，这样就得到用于降维的转换矩阵W。这里有一点需要注意的是W降维的大小不能超过k-1即数据类别数-1。因为矩阵的秩小于等于各个相加得到它的矩阵的秩的和，而累加得到$$S_{b}$$的$$(\mu_{j}-\mu)(\mu_{j}-\mu)^T$$的秩为1，所以$$S_{b}$$的秩不超过k，又因为第k个$$\mu_{k}-\mu$$可由前k-1个$$\mu_{j}-\mu$$线性表示，因此$$S_b$$秩最大为k-1，则不为0的特征值最多有k-1个。

### 1.2 LDA vs PCA

* Commonalities：
  * 两者均可以对数据进行降维。
  * 两者在降维时均使用了矩阵特征分解的思想。
  * 两者都假设数据符合高斯分布。
* Differences：
  * LDA是有监督的降维方法，而PCA是无监督的降维方法
  * LDA降维最多降到类别数==k-1==的维数，而PCA没有这个限制。
  * LDA除了可以用于降维，还可以用于分类。
  * LDA选择分类性能最好的投影方向，而PCA选择样本点投影具有最大方差的方向。


## 二、Neighborhood Component Analysis(NCA)

### 2.1 Rationale

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;近邻成分分析（NCA）是由Jacob Goldberger和Geoff Hinton等大佬们在2005年发表于NIPS上的一项工作[^1]，属于度量学习（Metric Learning）和降维（Dimension Reduction）领域。NCA的原理是以马氏距离为距离度量的KNN为基础，通过不断优化KNN分类的准确率来学习转换矩阵，最终得到对原数据进行降维的转换矩阵。

[^1]: <http://www.cs.toronto.edu/~fritz/absps/nca.pdf>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;接下来对NCA学习转换矩阵的数学推导，设$$x_i(1 \le i \le n)$$表示原数据的列向量表示，A是d*D的转换矩阵，其中D为原数据的维度，而d为降维后的维度，定义$$p_{ij}$$为==映射空间中欧氏距离==(相当于原空间中的马氏距离)的softmax概率值
$$
p_{ij}=\frac{exp(- \|Ax_{i}-Ax_{j}\|^{2})}{\sum_{k \ne i}exp(-\|Ax_{i}-A_{k}\|)}，\ p_{ii}=0
$$
设$p_{i}$为i能够被正确分类的概率，$C_{i}$表示与i属于同一类的样本的集合，那么
$$
p_{i}=\sum_{j \in C_{i}}p_{ij}
$$
优化目标就是最大化能被正确分类的点的数目
$$
f(A)=\sum_{i}\sum_{j \in C_{i}}p_{ij}=\sum_{i}p_{i}
$$
f(A)对A求偏导，定义$$x_{ij}=x_{i}-x_{j}$$
$$
\begin{aligned}
f^{'}(A)=\frac{\partial f}{\partial A}&=-2A\sum_{i}\sum_{j \in C_{i}}p_{ij}(x_{ij}x_{ij}^{T}-\sum_{k}p_{ik}x_{ik}x_{ik}^T)\\
&=2A\sum_{i}(p_{i}\sum_{k}p_{ik}x_{ik}x_{ik}^T-\sum_{j \in C_{i}}p_{ij}x_{ij}x_{ij}^T)
\end{aligned}
$$
有了目标函数对A梯度之后就可以设定迭代次数和A的初始值$$A_{0}$$，利用梯度下降法不断优化目标函数上限(当然也可以使用其它的优化方法比如拟牛顿法)，设学习率为$$\alpha$$，$$A_{0}$$通过下面公式迭代
$$
A_{0}=A_{0}+\alpha f^{'}(A_{0})
$$

### 2.2 NCA vs PCA

* commonalities:
  * 都可以用来降维
* differences:
  * NCA除了降维还是一种度量学习的方法
  * NCA对数据分布没有假设，而PCA要求数据服从高斯分布
  * NCA基于KNN选择分类性能最好的投影方向，而PCA选择样本点投影具有最大方差的方向