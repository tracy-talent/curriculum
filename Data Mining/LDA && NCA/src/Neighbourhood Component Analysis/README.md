fork from:[NCA](https://github.com/lxcnju/Neighbourhood-Component-Analysis)
# Neighbourhood-Component-Analysis
Four methods to implementing NCA which is often used for metric learning and dimension reduction.


* 知乎链接
* 代码架构
* 方法
  * 两层for循环
  * 矩阵操作加速
  * 另一种梯度表达形式
  * scipy.optimize实现
* 结果展示

## 知乎(zhihu)链接
  * [Neighbourhood Component Analysis ](https://zhuanlan.zhihu.com/p/48371593)

## 代码架构
 * nca_naive.py  使用两层for循环求梯度的实现；速度很慢
 * nca_matrix.py 使用矩阵操作加速，但是仍然有一层for循环；速度稍微快了一点，但是占内存
 * nca_fast.py   使用梯度的另一种形式，全矩阵操作，没有for循环，速度很快，空间占用少
 * nca_scipy.py  使用nca_fast.py的方法 + scipy.optimize的优化包实现，分为gradient descent和coordinate descent
 * example.py    利用mnist数据集进行测试
 * usage.py      里面展示了使用方法，由于四种实现都封装为了NCA类，并且实现了类似PCA的fit, fit_transform, transform方法，使用很简单

## 方法
  * 两层for循环
  * 矩阵操作加速
  * 另一种梯度表达形式
  * scipy.optimize实现

## 结果展示
  下面给出一些结果图片,前面9张是在mnist上面的结果，选取的数字类别数目分别为2至9，由于原图是784维的，所以先使用PCA降维到100维，然后再使用NCA降维到2维；后面三张是直接使用NCA在digits(numpy提供)，breast_cancer和iris数据集降维到2维上得到的结果。
  <div> 
    <table>
     <tr>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/mnist_with_2_digits.jpg"></td>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/mnist_with_3_digits.jpg"></td>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/mnist_with_4_digits.jpg"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/mnist_with_5_digits.jpg"></td>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/mnist_with_6_digits.jpg"></td>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/mnist_with_7_digits.jpg"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/mnist_with_8_digits.jpg"></td>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/mnist_with_9_digits.jpg"></td>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/mnist_with_10_digits.jpg"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/digits_np.jpg"></td>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/breast_cancer.jpg"></td>
      <td><img src = "https://github.com/lxcnju/Neighbourhood-Component-Analysis/blob/master/pics/iris.jpg"></td>
     </tr>
     
    </table>
  </div>



