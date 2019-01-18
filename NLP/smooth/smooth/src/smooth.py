from pyltp import Segmentor
from pyltp import Postagger

import numpy as np

import re
import os
import time
import math
import random

'''
基于2-Grams模型判断句子通顺与否
'''
print('开始时间：', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# 分词器与词性标注器
timestamp = time.time()
print('加载pyltp模型...')
seg = Segmentor()
seg.load(os.path.join('../input/ltp_data_v3.4.0', 'cws.model'))
pos = Postagger()
pos.load(os.path.join('../input/ltp_data_v3.4.0', 'pos.model'))
print('耗时：', time.time() - timestamp, 's')

timestamp = time.time()
print('预处理训练数据...')
train_sent = []
trainid = []
train_sent_map = {}  #[id:[ngram value, label]]
with open('../input/train.txt', 'r') as trainf:
    for line in trainf:
        if len(line.strip()) != 0:
            items = line.strip().split('\t')
            tags = ['<s>']
            for tag in pos.postag(seg.segment(items[1])):
                tags.append(tag)
            tags.append('</s>')
            train_sent.append(tags)
            trainid.append(int(items[0]))
            train_sent_map[trainid[-1]] = [0.0, int(items[2])]
print('耗时:', time.time() - timestamp, 's')

timestamp = time.time()
print('预处理测试数据...')
test_sent = []
testid = []
with open('../input/test.content.txt', 'r') as testf:
    for line in testf:
        if len(line.strip()) != 0:
            items = line.strip().split('\t')
            tags = ['<s>']
            for tag in pos.postag(seg.segment(items[1])):
                tags.append(tag)
            tags.append('</s>')
            test_sent.append(tags)
            testid.append(items[0])  
print('耗时：', time.time() - timestamp, 's')


# 训练
timestamp = time.time()
print('训练数据特征处理...')
train_1gram_freq = {}
train_2grams_freq = {}
for sent in test_sent:
    train_1gram_freq[sent[0]] = 1
    for j in range(1, len(sent)):
        train_1gram_freq[sent[j]] = 1
        train_2grams_freq[' '.join(sent[j-1:j+1])] = 1e-100

for i in range(len(trainid)):
    if train_sent_map[trainid[i]][1] == 0:
        sent = train_sent[i]
        train_1gram_freq[sent[0]] = 0
        for j in range(1, len(sent)):
            train_1gram_freq[sent[j]] = 0
            train_2grams_freq[' '.join(sent[j-1:j+1])] = 0

# 预处理训练集0标记的正确句子
for i in range(len(trainid)):
    if train_sent_map[trainid[i]][1] == 0:
        sent = train_sent[i]
        train_1gram_freq[sent[0]] += 1
        for j in range(1, len(sent)):
            train_1gram_freq[sent[j]] += 1
            train_2grams_freq[' '.join(sent[j-1:j+1])] += 1
print('耗时：', time.time() - timestamp, 's')

def compute_2grams_prob(sent, train_1gram_freq, train_2grams_freq):
    p = 0.0
    for j in range(1, len(sent)):
        p += math.log(train_2grams_freq[' '.join(sent[j-1:j+1])] * 1.0  \
            / train_1gram_freq[sent[j-1]])
    return p / len(sent)

timestamp = time.time()
print('训练N-Gram模型...')
# 插值权值
for i, sent in enumerate(train_sent):
    if train_sent_map[trainid[i]][1] == 0:
        train_sent_map[trainid[i]][0] = compute_2grams_prob(sent, train_1gram_freq, train_2grams_freq)

train_samesize_avgprob = {}
for i, sent in enumerate(train_sent):
    train_samesize_avgprob[len(sent)] = [0.0, 0]
for i, sent in enumerate(train_sent):
    train_samesize_avgprob[len(sent)][0] += train_sent_map[trainid[i]][0]
    train_samesize_avgprob[len(sent)][1] += 1
for key in train_samesize_avgprob.keys():
    train_samesize_avgprob[key][0] = train_samesize_avgprob[key][0] / train_samesize_avgprob[key][1]

thresh = {'min':0.0, 'max':-np.inf, 'avg':0.0}
c0 = 0
for id in trainid:
    if train_sent_map[id][1] == 0:
        if train_sent_map[id][0] < thresh['min']:
            thresh['min'] = train_sent_map[id][0]
        if train_sent_map[id][0] > thresh['max']:
            thresh['max'] = train_sent_map[id][0]
        thresh['avg'] += train_sent_map[id][0]
        c0 += 1
thresh['avg'] /= c0
print('训练集：', thresh)
print('耗时：', time.time() - timestamp, 's')


# 预测
timestamp = time.time()
print('预测...')
thresh_tx = {'min':0.0, 'max':-np.inf, 'avg':0.0}
TX = []
with open('../output/result.txt', 'w') as resf:
    for sent in test_sent:
        TX.append([compute_2grams_prob(sent, train_1gram_freq, train_2grams_freq)])
    for i in range(len(TX)):
        if thresh_tx['min'] > TX[i][0]:
            thresh_tx['min'] = TX[i][0]
        if thresh_tx['max'] < TX[i][0]:
            thresh_tx['max'] = TX[i][0]
        thresh_tx['avg'] += TX[i][0]
    thresh_tx['avg'] /= len(TX)
    print('测试集：', thresh_tx)
    for i in range(len(testid)):
        if len(test_sent[i]) in train_samesize_avgprob.keys():
            if TX[i][0] >= train_samesize_avgprob[len(test_sent[i])][0]:
                resf.write(testid[i] + '\t0\n')
            else:
                resf.write(testid[i] + '\t1\n')
        else:
            if TX[i][0] >= thresh['avg'] - 0.1:
                resf.write(testid[i] + '\t0\n')
            else:
                resf.write(testid[i] + '\t1\n')
print('耗时：', time.time() - timestamp, 's')

print('结束时间：', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))