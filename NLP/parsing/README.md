<center><h1>parsing</h1></center>



## 一、任务要求

* 实现一个基于简单英语语法的chart句法分析器。



## 二、技术路线

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;采用自底向上的句法分析方法，简单的自底向上句法分析效率不高，常常会重复尝试相同的匹配操作（回溯之前已匹配过）。一种基于图的句法分析技术（Chart Parsing）被提出，它把已经匹配过的结果保存起来，今后需要时可直接使用它们，不必重新匹配。（动态规划）

- chart parsing的数据表示：
  - p图（chart）的结点表示句子中词之间的位置数字
  - p非活动边集（chart的核心，常直接就被称为chart
    - n记录分析中规约成功所得到的所有词法/句法符号
  - 活动边集
    - 未完全匹配的产生式，用加小圆圈标记（º）的产生式来表示，如：
      - NP -> ART ºADJ N
      - NP -> ART ºN
  - 待处理表（agenda）
    - 实际上是一个队列模型，记录等待加入chart的已匹配成功的词法/句法符号
  - 上面的活动边、非活动边以及词法/句法符号都带有“始/终结点”位置信息
- chart parsing对“~1~ The ~2~ cat ~3~ caught ~4~ a ~5~ mouse ~6~”进行分析的数据示例：

![1541994559204](https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/nlp_parsing_1.png)

* chart parsing的句法分析算法步骤描述如下：
  * 若agenda为空，则把句子中下一个词的各种词法符号（词）和它们的位置加入进来
  * 从agenda中取一个元素（设为C，位置为：p1-p2)
  * 对下面形式的每个规则增加活动边：
    * X->CX~1~...X~n~，增加一条活动边：X->C º X~1~...X~n~，位置为：p1-p2；
    * X->C，把X加入agenda，位置为：p1-p2
  * 将C作为非活动边加入到chart的位置p1-p2
  * 对已有活动边进行边扩展
    * 对每个形式为：X->X~1~... º C...X~n~的活动边，若它在p0-p1之间，则增加一条活动边：X->X~1~... C º...X~n~，位置:p0-p2
    * 对每个形式为： X->X~1~... X~n~ º C的活动边，若它在p0-p1之间，则把X加入agenda ，位置为：p0-p2

* 程序实现的大致流程：输入英文语句，对在词典dic_ec.txt中不存在的英文单词进行形态还原，对还原后的语句执行chart parsing算法并将分析出的所有非活动边输出。由于一个英文单词可能存在多种词性，这种情况下会对每种可能的词性进行递归，对于不符合句法规则的词性会进行回溯尝试以其它的词性进行句法规则的匹配与分析。直到找到符合句法规则的词性组合则结束递归，尝试完所有的词性组合还是没能找到则句法分析失败，输入的句子不符合当前的句法规则。



## 三、数据说明

* 由于这个实验中引用了token实验模块，所以需要用到token实验中的三个数据字典dic_ec.txt，irregualr nouns.txt，irregular verbs.txt，关于这三个数据字典的说明在token实验中已给出，此处不再赘述。除此之外，chart parsing算法还需要用到dic_ec.txt词典中英文单词的词性。



## 四、遇到的问题及解决方案

* 程序实现过程中受到文件编码和分隔符的困扰，最后用vim把用到的3个数据词典统一设置成gbk编码，以\t进行分隔，方便程序统一读入处理。
* dic_ec.txt这个数据字典中的数据质量不太好，很多英文单词都被标注成none.词性，由于无法获取词的正确词性从而无法完成句子的句法分析。



## 五、性能分析

* 对句法分析部分作一个性能的度量，单句句法分析的结果基本都在毫秒级别，下面给出基于规则S->NP VP，NP->ART N，NP->ART ADJ N，VP->V，VP->V NP对the cat catch a mouse进行句法分析得到的运行结果及耗时截图：

![1542023773939](https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/nlp_parsing_2.png)



## 六、运行环境

* 将执行文件parsing.exe与数据字典dic_ec.txt，irregular nouns.txt，irregualr verbs.txt放在同一个文件夹下，然后点击parsing.exe即可正常运行程序。



## 七、用户手册

* 在运行环境下正常运行程序后会出现下图这样的主菜单文字界面：

![1542024516293](https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/nlp_parsing_3.png)

* 根据主菜单进行操作，首先选择1来写入规则，可一次写入多个规则，输入q!结束规则写入，如果后期需要增加规则，可以在主菜单界面再次选择1来写入增添的规则，这样就实现了规则的灵活扩展。下面是写入规则模块的截图：

![1542025005658](https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/nlp_parsing_4.png)

* 写入规则结束后又回退到主菜单界面，这时候可以选择2来输入句子进行句法分析，程序会输出分析过程中得到的所有非活动边对应的短语及位置范围，下面是在上面所写入规则的基础上对the cat caught a mouse进行句法分析的结果截图，程序会对dic_ec.txt中不存在的单词尝试调用词形还原模块进行还原再分析：

![1542026424475](https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/nlp_parsing_5.png)

* 句法分析回退到主菜单界面，可以继续选择1进行规则扩展，也可以选择2进行句法分析，选择q退出程序运行。
* 基于下面的句法规则给出一个句法分析示例：

```
NP->ART N
NP->ART
NP->PRON
NP->N
NP->ART ADJ N
VP->V
VP->V NP
```

对I like her进行句法分析的结果截图如下：

![1542031147205](https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/nlp_parsing_6.png)



