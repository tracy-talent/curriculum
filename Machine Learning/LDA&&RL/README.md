## 一、LDA实验

### 1.1 文件说明

LDA相关内容全部放在LDA文件夹下，如下图

<div align="center">
    <img src="https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/ML_LDA_DIR.png">
</div>

其中corpus存放训练数据，stopword_en.txt是英文停词文件，src里面放的是源代码，其中任务1使用LDA我是用python调用sklearn写的，放在src的python目录下，任务2自己实现LDA我是用java写的，放在src的java目录下，nlp-1.0.jar是其对应的可执行jar包。



### 1.2 How to Launch

* LDA借助sklearn的python实现：

  > 系统环境：linux
  >
  > python环境：python3.6，scikit-learn 0.20.2，nltk 3.4，nltk packages：averaged_perceptron_tagger，punkt，stopwords

  进入到LDA/src/python目录下，首先在Terminal下进入到python命令行环境，执行以下命令：

  ```python
  import nltk
  nltk.download()
  ```

  通过上面的命令下载nltk的3个package:averaged_perceptron_tagger，punkt，stopwords，接下来就可以在Terminal下执行LDA

  ```shell
  python sklearnLDA.py 5
  ```

  sklearnLDA.py运行需要传入一个命令行参数：主题数，得到的结果保存在LDA/output_python目录下，如果想一次性得到主题5,10,20的结果，可以在Terminal下执行

  ```shell
  python integrated.py
  ```

  运行结束后得到的结果同样保存在LDA/output_python目录下
* LDA不借助开源的python实现：

  > 系统环境：linux
  >
  > python环境：python3.6，nltk 3.4，nltk packages：averaged_perceptron_tagger，punkt，stopwords

  进入到LDA/src/python目录下，首先在Terminal下进入到python命令行环境，执行以下命令：

  ```python
  import nltk
  nltk.download()
  ```

  确保nltk的3个package:averaged_perceptron_tagger，punkt，stopwords，接下来就可以在Terminal下执行myLDA.py

  ```python
  python myLDA.py
  ```

  第一次执行会将输入数据分词结果保存在LDA/corpus/segwords.txt中，以后执行将就直接从分词文件中读入数据从而缩减运行时间。运行结束后的结果保存在LDA/output_python/mylda目录下

* LDA不借助开源的java实现：

  > 系统环境：linux
  >
  > java环境：jdk1.8版本，Stanford coreNLP 3.9.2

  这部分的源代码放在LDA/src/java目录下，其中segment目录是分词模块，LDA目录是主程序模块。由于相比python的pip安装，java的maven安装没那么便捷，所以我把程序编译结果和依赖打包成为一个jar包nlp-1.0.jar方便运行，只需要有java环境即可，Terminal下进入到LDA目录下，执行

  ```shell
  java -cp nlp-1.0.jar LDA.TestLDA_ES 5
  ```

  同sklearnLDA.py一样，LDA.TestLDA_ES运行需要传入一个命令行参数：主题数，运行结束后得到的结果保存在LDA/output目录下(git上传100M限制，jar包就不上传了)



## 二、强化学习实验

### 2.1 文件说明

强化学习实验相关内容全部放在RL目录下，如下图

<div align="center">
    <img src="https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/ML_RL_DIR.png">
</div>

其中assets目录下是flappy bird游戏的音频和图片资源，game目录下是游戏代码，saved_model目录下是模型训练后保存的模型参数和checkpoint，deep_q_network.py是完善后的代码，Flappy Bird.gif是结果录制。



### 2.2 How to Launch

> 系统环境：linux
>
> python环境：python3，pygame 1.9.4，opencv-python 3.4.4，tensorflow 1.12

满足上述环境要求后在Terminal下进入到RL目录中，执行

```shell
python deep_q_network.py
```

由于save_model目录下保存了我训练300w轮的模型，并且程序中INITIAL_EPSILON和FINAL_EPSILON被设为一致了，所以程序运行后模型不会再训练学习，而是直接跑存好的模型结果，因此不出意外小鸟应该能稳健飞行，飞行的gif动图放在RL/Flappy Bird.gif中。



