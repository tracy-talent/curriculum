import time
import sys
import json

import pyfpgrowth

from config import *

sys.setrecursionlimit(10000000)

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
                data.append(line[line.index('{') + 1 : -3].split(','))
    elif dataset_path_name == 'UNIX_usage':
        itemset = []
        for subdir in os.listdir(UNIX_PATH):
            if subdir not in filter_subdir:
                continue
            subdir_path = UNIX_PATH + '/' + subdir
            if os.path.isdir(subdir_path):
                for subfile in os.listdir(subdir_path):
                    with open(subdir_path + '/' + subfile, 'r') as infile:
                        for line in infile:
                            if line.strip() == '**SOF**':
                                itemset = []
                            elif line.strip() == '**EOF**':
                                data.append(itemset)
                            else:
                                itemset.append(line.strip())


def output(freq_patterns, association_rules, dataset_path_name):
    if not os.path.exists(os.path.join(OUTPUT_PATH, dataset_path_name + '_fpgrowth')):
        os.makedirs(os.path.join(OUTPUT_PATH, dataset_path_name + '_fpgrowth'))
    freqfile = open(os.path.join(OUTPUT_PATH, dataset_path_name + '_fpgrowth/freq_patterns.txt'), 'w')
    for key, value in freq_patterns.items():
        freqfile.write(','.join(key) + ': ' + str(value) + '\n')
    freqfile.close()
    #写入关联规则
    association_file = open(os.path.join(OUTPUT_PATH, dataset_path_name + '_fpgrowth/association_rules.txt'), 'w')
    for key,value in association_rules.items():
        association_file.write(','.join(key) + ' -> ' + ','.join(value[0]) + '\tconfidence: ' + str(value[1]) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python fpgrowth.py dataset-path support-ratio confidence-ration')
    else:
        dataset_path_name = sys.argv[1]
        min_sup = float(sys.argv[2])  # 支持率
        min_con = float(sys.argv[3])
        print(min_con)
        data = []
        read(data, dataset_path_name)
        frequency_threshold = int(len(data) * min_sup)
        print("支持率对应项集频次阈值:", frequency_threshold)

        freq_patterns = pyfpgrowth.find_frequent_patterns(data, frequency_threshold)
        association_rules = pyfpgrowth.generate_association_rules(freq_patterns, min_con)
        output(freq_patterns, association_rules, dataset_path_name)
