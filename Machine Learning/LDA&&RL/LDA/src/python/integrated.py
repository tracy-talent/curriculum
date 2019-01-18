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
    n_top_words = 10
    expected_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", \
                            "NN", "NNS", "NNP", "NNPS", \
                            "JJ", "JJR", "JJS"]
    swlist = []
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
    tf_vectorizer = CountVectorizer(stop_words="english", lowercase=False)
    word_freq = tf_vectorizer.fit_transform(corpus)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print("语料库总词数：", len(tf_feature_names))
    lda = LatentDirichletAllocation(max_iter=50, doc_topic_prior=0.5, \
        topic_word_prior=0.1, learning_method="batch", random_state=0)
    for n_topics in [5, 10 ,20]:
        lda.set_params(n_components =  n_topics)
        params = lda.get_params(False)
        print("\nLDA模型参数：")
        for key ,value in params.items():
            print(key, "<-", value)
        timestamp = time.time()
        print("LDA(" + "n_components = " +str(n_topics) + ")训练...")
        lda.fit(word_freq)
        print("LDA(" + "n_components = " +str(n_topics) + ")训练耗时：", time.time() - timestamp, "s")
        print("输出结果到" + "../../output_python/topic" + str(n_topics) + "/topic-top"+str(n_top_words)+"keywords.txt")
        save_top_topciwords(lda, tf_feature_names, n_top_words)
    print("\n结束时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
