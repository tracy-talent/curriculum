import time
import os
import sys
import copy
from config import *

def read(data, dataset_path_name):
    """
    读取事务
    :param data: 存储事务的列表
    :param dataset_path_name: 命令行传入的要读取的数据集目录名
    :return: None
    """
    if dataset_path_name == 'GroceryStore':
        with open(os.path.join(GROCERY_PATH, 'Groceries.csv'), 'r') as infile:
            infile.readline()
            for line in infile:
                data.append(set(line[line.index('{') + 1 : -3].split(',')))
    elif dataset_path_name == 'UNIX_usage':
        itemset = set()
        for subdir in os.listdir(UNIX_PATH):
            subdir_path = UNIX_PATH + '/' + subdir
            if os.path.isdir(subdir_path):
                for subfile in os.listdir(subdir_path):
                    with open(subdir_path + '/' + subfile, 'r') as infile:
                        for line in infile:
                            if line.strip() == '**SOF**':
                                itemset = set()
                            elif line.strip() == '**EOF**':
                                data.append(itemset)
                            else:
                                itemset.add(line.strip())


def output(freqset, k_item_freq, frequency_threshold, k, dataset_path_name):
    """
    输出k频繁项集
    :param freqset: k频繁项集列表
    :param k_item_freq: k频繁项集字典{itemset:freq}
    :param frequency_threshold: 支持率对应的频次阈值
    :param k: k频繁项集的k, 用于构造输出文件名
    :param dataset_path_name: 命令行传入的要读取的数据集目录名
    :return: None
    """
    if not os.path.exists(os.path.join(OUTPUT_PATH, dataset_path_name + '_advanced_1')):
        os.makedirs(os.path.join(OUTPUT_PATH, dataset_path_name + '_advanced_1'))
    with open(os.path.join(OUTPUT_PATH, dataset_path_name + '_advanced_1/freq-' + str(k) + '-itemsets.txt'), 'w') as outfile:
        for key in freqset:
            keystr = ','.join(key)
            if k_item_freq[keystr] >= frequency_threshold:
                outfile.write(keystr + ' : ' + str(k_item_freq[keystr]) + '\n')


def prune(data, prune_basis, k):
    """
    由k-1的频繁项集生成k候选项集
    :param data: 存储事务的列表
    :param prune_basis: 由k候选项集匹配事务表时生成的剪枝依据
    :param k: 为生成k+1频繁项集剪枝， 剪枝依据list
    :return: None
    """
    if len(prune_basis) != 0:
        h = 0
        for i in range(len(prune_basis)):
            if prune_basis[i] < k + 1:
                del data[h]
            else:
                h += 1


def generate_candidates(pre_freqset, k):
    """
    由k-1的频繁项集生成k候选项集
    :param pre_freqset: 前一次生成的k-1频繁项集（项内与项之间都为有序状态）
    :param k: 要生成的k频繁项集对应的k
    :return: 包含k候选项集的列表
    """
    candidates = []
    if k == 2:
        for i in range(len(pre_freqset)):
            for j in range(i + 1, len(pre_freqset)):
                candidates.append([pre_freqset[i][0], pre_freqset[j][0]])
    else:
        i = 0
        while i < len(pre_freqset) - 1:
            tails = []
            while i < len(pre_freqset) - 1 and pre_freqset[i][:-1] == pre_freqset[i + 1][:-1]:
                tails.append(pre_freqset[i][-1])
                i += 1
            if tails:
                tails.append(pre_freqset[i][-1])
                prefix = copy.deepcopy(pre_freqset[i][0:-1])
                for a in range(len(tails)):
                    for b in range(a + 1, len(tails)):
                        items = copy.deepcopy(prefix)
                        items.append(tails[a])
                        items.append(tails[b])
                        candidates.append(items)
            i += 1

    return candidates


def count_freqset(data, candidates_k):
    """
    对照事务表统计候选项集频次
    :param data: 事务集
    :param candidates_k: k候选项集
    :return: 候选项映射到频次的dict
    """
    prune_basis = []
    for i in range(len(data)):
        prune_basis.append(0)
    counter = {}
    for can in candidates_k:
        canset = set(can)
        canstr = ','.join(can)
        counter[canstr] = 0
        for i in range(len(data)):
            if canset <= data[i]:
                counter[canstr] += 1
                prune_basis[i] += 1
    return counter, prune_basis


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python apriori_advanced.py dataset-path support-ratio K')
    else:
        dataset_path_name = sys.argv[1]
        min_sup = float(sys.argv[2])  # 支持率
        itemset_size = int(sys.argv[3])  # 最大频繁项集大小
        
        data = []
        # 读取数据集,每行记录以集合形式存储在data中
        read(data, dataset_path_name)
        # 对应支持率的频次
        frequency_threshold = int(len(data) * min_sup)
        print("支持率对应项集频次阈值:", frequency_threshold)
        # 生成频繁一项集
        start_time = time.time()
        oneitem_freq = {}
        for itemset in data:
            for item in itemset:
                oneitem_freq[item] = 0
        for itemset in data:
            for item in itemset:
                oneitem_freq[item] += 1
        oneitem_freqset = []
        for oneitem in oneitem_freq.keys():
            if oneitem_freq[oneitem] >= frequency_threshold:
                oneitem_freqset.append([oneitem])
        print('generate', len(oneitem_freqset), 'Frequent 1-Item Set waste time', time.time() - start_time, 's.')
        # 输出频繁项集到指定路径下
        oneitem_freqset.sort()
        output(oneitem_freqset, oneitem_freq, frequency_threshold, 1, dataset_path_name)
        pre_freqset = oneitem_freqset
        for k in range(2, itemset_size + 1):
            start_time = time.time()
            candidates_k = generate_candidates(pre_freqset, k)
            k_item_freq, prune_basis = count_freqset(data, candidates_k)
            prune(data, prune_basis, k)
            pre_freqset = []
            new_k_item_freq = {}
            for key in k_item_freq.keys():
                if k_item_freq[key] >= frequency_threshold:
                    pre_freqset.append(key.split(','))
                    new_k_item_freq[','.join(pre_freqset[-1])] = k_item_freq[key]
            k_item_freq = new_k_item_freq
            if len(pre_freqset) == 0:
                break
            pre_freqset.sort()
            print('generate', len(pre_freqset), 'Frequent', str(k) + '-Item Set waste time', time.time() - start_time, 's.')
            output(pre_freqset, k_item_freq, frequency_threshold, k, dataset_path_name)
