<center><h1>seg</h1></center>



## 一、任务要求

* 实现一个基于词典与规则的汉语自动分词系统。



## 二、技术路线

* 采用正向最大匹配(FMM)方法对输入的中文语句进行分词，具体的实现可以分为下面几个步骤：
  1. 对输入的一个中文语句，首先在程序中判断并确保语句中不包含数字或者字母
  2. 在句子中的当前位置开始取与词典dic_ce.txt中最大匹配长度的词作为一个分词段，如果没有在词典中成功匹配到就将句子在当前匹配位置的这个字作为一个分词段并将匹配位置向前挪一个位置
  3. 重复第2步直到匹配位置移到句末

* 下面是用FMM方法分词的具体实现:

```c++
//param@seg:保存分词段结果的vector
//param@st:带分词的中文语句
void segment(vector<string> &seg, string st) {
    int pos = 0;
    int sz = st.length();
    string t;
    int cnt = 0, spos;
    while (pos < sz) {
        cnt = pos;
        spos = pos;
        t = "";
        while (st[cnt]) {
            t += st.substr(cnt, 2);
            if (wordmap.find(t) != wordmap.end())
                pos = cnt + 2;
            cnt += 2;
        }
        if (pos == spos) {
            seg.push_back(st.substr(spos, 2));
            pos += 2;
        }else {
            seg.push_back(st.substr(spos, pos - spos));
        }
    }
    return;
}
```



## 三、数据说明

* 汉英词典dic_ce.txt，读取其中的汉词用于与句中词进行匹配，词典采用GBK编码，下面是给出文件内容示例：

```
//gbk编码，每行第一个词是汉词，后面是它对应的英译单词，以','分隔
阿弥陀佛,Amitabha
阿米巴,amoeba,amoebae
阿姆斯特丹,Amsterdam
阿斯匹林,aspirin
```



## 四、性能分析

* 假设输入中文语句长度为n，程序时间复杂度最坏情况下是O(n^2)，最好情况是O(n)，下面是程序分析结果及分词耗时评测的截图：

![1541992901499](https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/nlp_seg_1.png)



## 五、运行环境

* 将执行文件seg.exe与数据字典dic_ce.txt放在同一个目录下，然后点击seg.exe即可正常运行，进入运行窗口后根据提示进行输入即可得到分词结果。