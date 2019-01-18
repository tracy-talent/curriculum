import os
import time
import random

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader import wordnet
from nltk.tag import pos_tag

# 保留的词性
expected_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", \
                        "NN", "NNS", "NNP", "NNPS", \
                        "JJ", "JJR", "JJS"]

# 加载停词表
fs=open('../../stopwords_en.txt')
stopwords = fs.read()
swlist = stopwords.splitlines()
fs.close()

print("step1：加载语料库及预处理")
timestamp = time.time()
corpus = []  #存放语料库，每个元素代表一篇文档
if not os.path.exists('../../corpus/segwords.txt'):
    lemmatizer = WordNetLemmatizer()
    with open('../../corpus/news.txt', 'r') as df:
        for line in df:
            if len(line.strip()) != 0:
                words = word_tokenize(line.strip())
                tags = pos_tag(words)
                seglist = []
                for i in range(len(words)):
                    if tags[i][1] in expected_tags and words[i] not in swlist and words[i].isalpha():
                        taghead = tags[i][1][0].lower()
                        # {ADJ:a, ADJ_SAT:s, ADV:r, NOUN:n or VERB:v} 词形还原
                        seglist.append(lemmatizer.lemmatize(words[i], wordnet.ADJ if taghead == 'j' else taghead))
                corpus.append(' '.join(seglist))
    with open('../../corpus/segwords.txt', 'w') as cf:
        for doc in corpus:
            cf.write(doc + '\n')
else:
    with open('../../corpus/segwords.txt', 'r') as cf:
        for line in cf:
            corpus.append(line.strip())
print('step1耗时', time.time() - timestamp, 's')

print('step2:建立词索引')
timestamp = time.time()
tokens = []
word_dict = {}
maxdoclen=0
for line in corpus:
    line = line.split(' ')
    if (len(line)>maxdoclen):
        maxdoclen = len(line)
    for word in line:
        word_dict[word]=0
    tokens.append(line)

wordid = -1
for key in word_dict:
    wordid+=1
    word_dict[key]=wordid
print('step2耗时:', time.time() - timestamp, 's')

print('最长文档词数:',maxdoclen)
vocasize=len(word_dict)
print('总词数:', vocasize)
docnum=len(tokens)
print('文档数:', docnum)
n_features = 10
iter_times = 100

print('step3: Gibbs sampling')
for K in [5, 10, 20]:
    print("%d topics"%K)
    timestamp = time.time()
    # LDA相关变量
    beta = 0.1
    alpha = 0.5
    p = [0.0] * K
    nw = [[0 for i in range(K)] for i in range(vocasize)]
    nwsum = [0] * K
    nd = [[0 for i in range(K)] for i in range(docnum)]
    ndsum = [0] * docnum
    z = [[(0, 0) for i in range(maxdoclen)] for i in range(docnum)]
    theta = [[0.0 for i in range(K)] for i in range(docnum)]
    phi = [[(0.0, 0) for i in range(vocasize)] for i in range(K)]

    # 初始化变量
    for i in range(docnum):
        ndsum[i] = len(tokens[i])
        j = -1
        for word in tokens[i]:
            j += 1
            topic_index = random.randint(0, K-1) #[0,K-1]
            word_id = word_dict[word]
            z[i][j] = (topic_index, word_id)
            nw[word_id][topic_index] += 1
            nd[i][topic_index] += 1
            nwsum[topic_index] += 1

    # gibbs sampling迭代计算
    Vbeta = vocasize * beta
    Kalpha = K * alpha
    for iter in range(iter_times):
        print('第 %d 轮迭代' % iter)
        ts = time.time()
        for i in range(docnum):
            for j in range(ndsum[i]):
                word_id = z[i][j][1]
                topic = z[i][j][0]
                nw[word_id][topic] -= 1
                nd[i][topic] -= 1
                nwsum[topic] -= 1
                for k in range(K):
                    p[k] = (nw[word_id][k] + beta) / (nwsum[k] + Vbeta) * (nd[i][k] + alpha) / (ndsum[i] + Kalpha)
                for k in range(1, K):
                    p[k] += p[k - 1]
                u = random.random() * p[K - 1]
                for t in range(K):
                    if p[t] > u:
                        break
                nw[word_id][t] += 1
                nwsum[t] += 1
                nd[i][t] += 1
                z[i][j] = (t, word_id)
        print('elpased', time.time() - ts, 's')

    # compute theta[d,z] and phi[z,w]
    for i in range(docnum):
        for j in range(K):
            theta[i][j] = (nd[i][j] + alpha) / (ndsum[i] + Kalpha)
    for i in range(K):
        for j in range(vocasize):
            phi[i][j] = ((nw[j][i] + beta) / (nwsum[i] + Vbeta) , j)
    print('topicNum = %d的' % K + 'Gibbs Sampling耗时', time.time() - timestamp, 's')

    print('输出topicNum = %d 结果：' % K)
    wordlist = list(word_dict.keys())
    if not os.path.exists('../../output_python/mylda'):
        os.makedirs('../../output_python/mylda')
    with open('../../output_python/mylda/topic' + str(K) + '.txt', 'w') as rf:
        for i in range(K):
            rf.write("Topic # %d:" % i + '\n')
            phi[i] = sorted(phi[i], key=lambda t: t[0], reverse=True)
            wordsize_phi = []
            for j in range(n_features):
                topicword = wordlist[phi[i][j][1]]
                wordsize_phi.append((len(topicword), (topicword, phi[i][j][0])))
            wordsize_phi = sorted(wordsize_phi, key=lambda t: t[0], reverse=True)
            for item in wordsize_phi:
                rf.write(item[1][0] + '\n')
                rf.write('%.2f' % (item[1][1]*100) + '%\n')
