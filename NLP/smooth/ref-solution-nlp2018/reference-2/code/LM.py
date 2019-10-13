# n-gram
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import math
import jieba

data = []


with open("./data/train.txt",encoding='UTF-8') as f:
    for i in range(109999):
        line = f.readline()
        line_split = line.split('\t')
        sentence = line_split[1]
        label = line_split[2]
        if int(label) == 1:
            sentence = "BEG" + sentence + "END"
            data.append(sentence)
data = [" ".join(jieba.lcut(e)) for e in data ] # 分词，并用" "连接
# ngram_range=(2, 2)表明适应2-gram,decode_error="ignore"忽略异常字符,token_pattern按照单词切割
ngram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                                        token_pattern = r'\b\w+\b',min_df=1)

x1 = ngram_vectorizer.fit_transform(data)
print(x1)
fre = x1.toarray()
fre_sum = np.sum(fre,axis=0)
print (type(x1.toarray()))
print(fre_sum)
# [[1 1 1 1 1 1 1 1 1 1]
#  [0 1 0 0 0 0 0 0 1 0]]
# 查看生成的词表
print(ngram_vectorizer.vocabulary_)
dic = ngram_vectorizer.vocabulary_
print(type(dic))
voc_Id = dic.get('他 用fs')
if voc_Id == None:
    print("sadsda")
frequence = fre_sum[voc_Id]
print(frequence)
fout = open("./data/MG1833031.txt", 'w', encoding='UTF-8')
with open("./data/test.txt",encoding='UTF-8') as f_test:
    for i in range(19990):
        line = f_test.readline()
        line_split = line.split('\t')
        id = line_split[0]
        sentence = line_split[1]
        sentence = sentence.replace('\n', '').replace('\t', ' ')
        sentence = "BEG"  + sentence +  "END"
        cut = jieba.lcut(sentence)
        cut_len = len(cut)
        print(cut)
        score = 0
        for j in range(cut_len-1):
            gram = cut[j]+" "+cut[j+1]
            voc_Id = dic.get(gram)
            frequent = 1
            if voc_Id != None:
                frequent += fre_sum[voc_Id]
            score += math.log(frequent)
        if score > 1:
            fout.writelines(id + '\t' + '1' + '\n')
        else:
            fout.writelines(id + '\t' + '0' + '\n')
            # print(gram+" "+str(frequent))

