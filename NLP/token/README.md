<center><h1>token</h1></center>



## 一、任务描述

- 形态还原算法：
  1. 输入一个单词
  2. 如果词典里有该词，输出该词及其属性，转4，否则，转3
  3. 如果有该词的还原规则，并且，词典里有还原后的词，则输出还原后的词及其属性，转4，否则，调用<未登录词模块>
  4. 如果输入中还有单词，转(1)，否则，结束。


## 二、技术路线

1. 加载dic_ec.txt词典，词典存储着英到汉的映射，对于输入的单词，如果dic_ec.txt词典中包含这个单词的映射则直接输出。下面给出dic_ec.txt内容的基本形式：

```
//gbk编码，以\t分隔
homokaryosis	none.	同核性, 同核现象
homokaryotic	adj.	同核体的
homokurtic	none.	等峰态性
homolanthionine	none.	高羊毛氨酸
```

2. 考虑到有些单词本身就是原形，也是其它单词的形态变换，所以在设计时决定把所有可能的结果都输出。在完成词典映射后再检查该单词是否能通过变换规则转换得到。我们知道英文单词的形态变换存在有规律的和无规律的变换，首先看有规律的变换，动词的规律变换形式有下面4条规则：

```
规则1.  *ves --> *f/*fe
规则2.  *ies --> *y
规则3.  *es  --> *
规则4.  *s   --> *
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;名次的规律变换形式有下面9条规则：

```
//第三人称单数
规则5.  *ies --> *y
规则6.  *es --> *
规则7.  *s   --> *
//现在进行时
规则8.  *??ing --> *?   
规则9.  *ying --> *ie
规则10.  *ing  --> */*e
//过去时、过去分词
规则11.  *??ed --> *?
规则12.  *ied --> *y
规则13.  *ed  --> */*e
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过在程序中写入这些规则来对单词形态进行还原，而无规则的形态变换只能通过预先建立好的词库来完成词形形态映射。在程序中通过加载irregualr nouns.txt对名词进行还原，加载irregualr verbs.txt对动词进行还原。下面分别给出这两文件中的内容形式：

&nbsp;&nbsp;irregular nouns.txt的内容形式：

```
//gbk编码，每行的第一个词是原形，后面的是变换形态，以\t分隔
grief	griefs
roof	roofs
gulf	gulfs
grief	griefs
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;irregualr verbs.txt的内容形式：

```
//gbk编码，每行的第一个词是原形，后面的是变换形态，以\t分隔
bear	bore	borne	born
alight	alighted	alit	alighted	alit
arise	arose	arisen
awake	awoke	awaked	awoken	awoke	awaked
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果找到了还原映射，则在dic_ec.txt词典中查找还原后的单词并输出结果。

3. 若最终该单词没有检索到结果则把他登记到单词缺失词典missing words.txt中。



## 三、数据说明

* 英汉词典dic_ec.txt，名词的不规律变换词典irregualr nouns.txt，动词的不规律变换词典irregualr verbs.txt，这几个数据词典的编码以及内容形式都已在技术路线中给出，此处不再赘述。



## 四、遇到的问题及解决方案

* 程序实现过程中唯一遇到的问题就是文件编码和分隔符的问题，最后用vim把用到的3个数据词典统一设置成gbk编码，以\t进行分隔，方便程序统一读入处理。



## 五、性能分析

* 下面是性能单词查询的耗时截图，平均不超过0.001s：

![1541928466093](https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/nlp_token_1.png)



## 六、运行环境

* 将token.exe与dic_ec.txt，irregualr nouns.txt，irregualr verbs.txt，missing words.txt放在同一个目录下，然后点击token.exe即可正确运行程序。
