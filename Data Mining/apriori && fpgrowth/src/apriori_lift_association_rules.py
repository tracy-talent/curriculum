import time
import os
import sys
import copy
import gc
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


def output(freqset, k_item_freq, association_rules, frequency_threshold, k, dataset_path_name, min_sup, min_con):
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
        outfile.write('支持度：' + str(min_sup) + ', 频次下界：' + str(frequency_threshold) + '\n')
        for key in freqset:
            keystr = ','.join(key)
            if k_item_freq[keystr] >= frequency_threshold:
                outfile.write(keystr + ' : ' + str(k_item_freq[keystr]) + '\n')
    if association_rules:
        if not os.path.exists(os.path.join(OUTPUT_PATH, dataset_path_name + '_lift_association_rules')):
            os.makedirs(os.path.join(OUTPUT_PATH, dataset_path_name + '_lift_association_rules'))
        with open(os.path.join(OUTPUT_PATH, dataset_path_name + '_lift_association_rules/' + str(k) + '-items-rules.txt'), 'w') as outfile:
            outfile.write('支持度：' + str(min_sup) + ', 置信度：' + str(min_con) + '\n')
            for item in association_rules:
                outfile.write('(' + item[0] + ') -> (' + item[1] + ')' + '\tconfidence:' + str(item[2]) + '\n')


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


def get_all_subsets(itemset):
    """
    计算itemset的所有势大于0的子集
    :param itemset: 包含一个项集的列表 
    :return: 项集itemset的所有势大于0的子集(type: list(set...))
    """
    n = len(itemset)
    all_subsets = []
    for i in range(1, 2**(n - 1)):
        subset = set()
        for j in range(n):
            if (i >> j) % 2 == 1:
                subset.add(itemset[j])
        all_subsets.append(subset)
    return all_subsets


def generate_association_rules(k_freqset, itemset_freq, min_con, dataset_size):
    """
    生成关联规则
    :param k_freqset: k频繁项集 
    :param itemset_freq: 项集与频次之间的字典映射
    :param min_con: 置信度
    :param dataset_size: 数据集记录总数
    :return: 关联规则列表(list(list(previous item, post item, frequency)...))
    """
    association_rules = []
    for i in range(len(k_freqset)):
        all_subsets = get_all_subsets(k_freqset[i])
        par_set = set(k_freqset[i])
        par_cnt = itemset_freq[','.join(k_freqset[i])]
        for subset in all_subsets:
            subset_str = ','.join(sorted(list(subset)))
            diffset_str = ','.join(sorted(list(par_set.difference(subset))))
            subset_cnt = itemset_freq[subset_str]
            diffset_cnt = itemset_freq[diffset_str]
            subset_prob = 1.0 * subset_cnt / dataset_size
            diffset_prob = 1.0 * diffset_cnt / dataset_size
            if 1.0 * par_cnt / subset_cnt >= min_con and 1.0 * par_cnt / (subset_cnt * diffset_prob) >= 1.0:
                association_rules.append([subset_str, diffset_str, 1.0 * par_cnt / subset_cnt])
            elif 1.0 * par_cnt / diffset_cnt >= min_con and 1.0 * par_cnt / (diffset_cnt * subset_prob) >= 1.0:
                association_rules.append([diffset_str, subset_str, 1.0 * par_cnt / diffset_cnt])

    return association_rules


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('usage: python apriori_advanced.py dataset-path support-ratio K confidence-ratio')
    else:
        dataset_path_name = sys.argv[1]
        min_sup = float(sys.argv[2])  # 支持率
        itemset_size = int(sys.argv[3])  # 最大频繁项集大小
        min_con = float(sys.argv[4])

        data = []
        # 读取数据集,每行记录以集合形式存储在data中
        read(data, dataset_path_name)
        # 对应支持率的频次
        dataset_size = len(data)
        frequency_threshold = int(len(data) * min_sup)
        print("支持率对应项集频次阈值:", frequency_threshold)
        # 生成频繁一项集
        start_time = time.time()
        oneitem_freq = {}
        itemset_freq = {}
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
                itemset_freq[oneitem] = oneitem_freq[oneitem]
        print('generate', len(oneitem_freqset), 'Frequent 1-Item Set waste time', time.time() - start_time, 's.')
        # 输出频繁项集到指定路径下
        oneitem_freqset.sort()
        output(oneitem_freqset, oneitem_freq, [], frequency_threshold, 1, dataset_path_name, min_sup, min_con)
        del oneitem_freq
        gc.collect()
        pre_freqset = oneitem_freqset
        for k in range(2, itemset_size + 1):
            start_time = time.time()
            candidates_k = generate_candidates(pre_freqset, k)
            k_item_freq, prune_basis = count_freqset(data, candidates_k)
            prune(data, prune_basis, k)
            del pre_freqset
            gc.collect()
            pre_freqset = []
            new_k_item_freq = {}
            for key in k_item_freq.keys():
                if k_item_freq[key] >= frequency_threshold:
                    pre_freqset.append(key.split(','))
                    new_k_item_freq[','.join(pre_freqset[-1])] = k_item_freq[key]
                    itemset_freq[','.join(pre_freqset[-1])] = k_item_freq[key]
            del k_item_freq
            gc.collect()
            k_item_freq = new_k_item_freq
            if len(pre_freqset) == 0:
                break
            pre_freqset.sort()
            association_rules = generate_association_rules(pre_freqset, itemset_freq, min_con, dataset_size)
            print('generate', len(pre_freqset), 'Frequent', str(k) + '-Item Set waste time', time.time() - start_time, 's.')
            output(pre_freqset, k_item_freq, association_rules, frequency_threshold, k, dataset_path_name, min_sup, min_con)
            del association_rules
            gc.collect()
