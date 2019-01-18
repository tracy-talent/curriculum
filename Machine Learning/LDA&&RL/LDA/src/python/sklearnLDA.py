from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader import wordnet
from nltk.tag import pos_tag

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import numpy as np

import re
import time
import os
import sys


# nltk.download()
# load stopwords
def load_stopwords(swlist):
    with open("../../stopwords_en.txt") as sw_dic:
        sw_content = sw_dic.read()
    for sw in sw_content.splitlines():
        swlist.append(sw)


# load news.txt
def load_corpus(expected_tags, lemmatizer, swlist, corpus):
    with open("../../corpus/news.txt", "r") as cf:
        for line in cf.readlines():
            if (len(line.strip()) != 0):
                words = word_tokenize(line.strip())
                tags = pos_tag(words)
                seglist = []
                for i in range(len(words)):
                    if tags[i][1] in expected_tags and words[i] not in swlist:
                        if len(re.sub("[^a-zA-Z]", "", words[i])) == len(words[i]):
                            taghead = tags[i][1][0].lower()
                            # {}ADJ:a, ADJ_SAT:s, ADV:r, NOUN:n or VERB:v}
                            seglist.append(lemmatizer.lemmatize(words[i], wordnet.ADJ if taghead == 'j' else taghead))
                corpus.append(" ".join(seglist))


# 保存模型训练结果
def save_top_topciwords(model, feature_names, n_top_words):
    OUTDIR = "../../output_python/topic" + str(model.get_params(False)["n_components"])
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    with open(os.path.join(OUTDIR, "topic-top"+str(n_top_words)+"keywords.txt"), "w") as resf:
        resf.write("\\begin{tabular}{c p{1.8cm}<{\\centering} p{1.6cm}<{\\centering} p{1.4cm}<{\\centering} p{1.4cm}<{\\centering} p{1.2cm}<{\\centering} p{1.2cm}<{\\centering} p{1.0cm}<{\\centering} p{1.0cm}<{\\centering} p{0.8cm}<{\\centering} p{0.8cm}<{\\centering}}\n")
        resf.write("\\toprule\n&$w1$&$w2$&$w3$&$w4$&$w5$&$w6$&$w7$&$w8$&$w9$&$w10$\\\\\n")
        for topic_idx, topicwords_phi in enumerate(model.components_ / model.components_.sum(axis=1)[:, np.newaxis]):
            wordlen_topicwords_phi = []
            for i in topicwords_phi.argpartition(-n_top_words)[:-(n_top_words + 1):-1]:
                wordlen_topicwords_phi.append((len(feature_names[i]), (i, topicwords_phi[i])))
            wordlen_topicwords_phi = sorted(wordlen_topicwords_phi, key=lambda t: t[0], reverse=True)
            resf.write("\\hline\n\\multirow{2}*{\\shortstack{t" + str(topic_idx+1) + "}}")            
            for item in wordlen_topicwords_phi:
                resf.write("&" + feature_names[item[1][0]])
            resf.write("\\\\\n")
            for item in wordlen_topicwords_phi:
                resf.write("&" + "%.2f" % (item[1][1]*100) + "\\%")
            resf.write("\\\\\n")
        resf.write("\\bottomrule\n")
        resf.write("\\end{tabular}")


if __name__ == "__main__":
    # 接收命令行传入的主题数目
    if (len(sys.argv) < 2):
        print("usage: python sklearnLDA.py n_topics")
        exit(1)
    n_topics = int(sys.argv[1])  
    # 输出的主题关键词数目
    n_top_words = 10  
    # 过滤的时候保留的词性
    expected_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", \
                            "NN", "NNS", "NNP", "NNPS", \
                            "JJ", "JJR", "JJS"]
    # 停词表
    swlist = []
    # 语料库
    corpus = []

    starttime = time.time()
    print('开始时间：', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    timestamp = time.time()
    print("加载停词表...")
    load_stopwords(swlist)
    print("加载停词表耗时：", time.time() - timestamp, "s")

    timestamp = time.time()
    print("分词...")
    lemmatizer = WordNetLemmatizer()
    load_corpus(expected_tags, lemmatizer, swlist, corpus)
    print("分词耗时：", time.time() - timestamp, "s")

    timestamp = time.time()
    print("词频统计...")
    tf_vectorizer = CountVectorizer(stop_words="english", lowercase=False)
    word_freq = tf_vectorizer.fit_transform(corpus)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print("语料库总词数：", len(tf_feature_names))
    print("词频统计耗时：", time.time() - timestamp, "s")

    timestamp = time.time()
    print("LDA训练...")
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50, doc_topic_prior=0.5, \
    topic_word_prior=0.1, learning_method="batch", random_state=0)
    lda.fit(word_freq)
    print("LDA训练耗时：", time.time() - timestamp, "s")

    # 输出主题下的n_top_words个关键词
    save_top_topciwords(lda, tf_feature_names, n_top_words)
    print("输出结果到" + "../../output_python/topic" + str(n_topics) + "/topic-top"+str(n_top_words)+"keywords.txt")
    print("结束时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))