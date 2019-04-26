>实验环境：
>
>python 3.6
>
>pyfpgrowth
>
>memory_profiler 
>
>psutil
>
>matplotlib



使用pip安装pyfpgrowth

```
pip install pyfpgrowth
pip install memory_profiler
pip install psutil
```



## 一、Apriori

* 频繁项集挖掘

  src目录下包含apriori频繁项集挖掘算法的三个版本: apriori_dummy.py，apriori_advanced_1.py，apriori_advanced_2.py，apriori_advanced_3.py分别对应实验报告第三节中的dummy apriori，advanced_1 apriori，advanced_2 apriori，advanced_3 apriori.下面以执行apriori_dummy.py为例，其他2个与此类似

  ```
  python apriori_dummy.py GroceryStore 0.01 10
  ```

  4个命令行参数：

  * 执行文件名.py
  * 数据集:GroceryStore或者UNIX_usage
  * 支持度
  * 需要挖掘的最大频繁项大小

  生成的频繁项集输出到output目录下


* 关联规则挖掘

  src目录下包含apriori关联规则挖掘算法的两个版本: apriori_association_rules.py，apriori_lift_association_rules.py，前者未使用提升度筛选，后者使用了提升度来筛选

  > 提升度公式定义
  > $$
  > lift(A->B) = \frac{P(AB)}{P(A)P(B)}
  > $$
  > 可以理解为向B推荐A的收益

  下面以执行apriori_association_rules.py为例，另一个与此类似

  ```shell
  python apriori_association_rules.py UNIX_usage 0.01 10 0.8
  ```

  5个命令行参数：

  * 执行文件名.py
  * 数据集：GroceryStore或者UNIX_usage
  * 支持度
  * 需要挖掘的最大频繁项大小
  * 置信度

  生成的频繁项集和关联规则输出到output目录下


## 二、Fpgrowth

* fpgrowth执行方式

  ```
  python fpgrowth.py GroceryStore 0.01 0.6
  ```

  4个命令行参数：

  - 执行文件名.py
  - 数据集:GroceryStore或者UNIX_usage
  - 支持度
  - 置信度

  生成的频繁项集和关联规则输出到output目录下



## 三、内存分析memory_profiler

用mprof run来替代python命令来执行.py文件

```shell
mprof run apriori_association_rules.py UNIX_usage 0.01 10 0.8
```

执行结束后将在当前目录下生成mprofile_xxxxxxxxxxx.data文件，里面存储了相同时间跨度的时间点上内存的使用情况，使用mprof list可以查看文件对应的索引(0,1,2...)，然后通过执行mprof plot file_index加载对应文件数据来绘制图形

```
mprof plot 0
```

