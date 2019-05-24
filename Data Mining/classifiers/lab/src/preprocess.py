import numpy as np
from collections import defaultdict

def load_arff(fpath):
    """
    加载读取.arff文件
     :param fpath: .arff文件路径
     :return: 包含样本属性和标签的二维数据矩阵data(numpy.array)
    """
    column_map = []
    data = []
    with open(fpath, 'r') as arff:
        for line in arff:
            if line[0] != '%' and line.strip() != '':
                if line.startswith('@attribute'):
                    if line.find('{') > 0:
                        items = line[line.find('{') + 1 : line.rfind('}')].split(',')
                        column_map.append({item.strip() : sid for sid, item in enumerate(items)})
                    else:
                        column_map.append(-1)
                elif line.strip()[0] != '@':
                    data.append([])
                    items = line.strip().split(',')
                    for i in range(len(items)):
                        if items[i] != '?' and column_map[i] != -1:
                            data[-1].append(column_map[i][items[i]])
                        else:
                            data[-1].append(items[i])
    return np.array(data)


def fill_miss(data):
    """
    使用同一类中出现次数最高的属性值来填补缺失值,直接修改data
     :param data: 样本数据矩阵numpy.array
     :return: X(数据属性np.array),Y(标签np.array)
    """
    hasmiss_2d = (data == '?')
    hasmiss_1d = hasmiss_2d.any(1)
    if hasmiss_1d.any():
        for i in range(data.shape[0]):
            if hasmiss_1d[i]:
                for j in range(data.shape[1]):
                    if hasmiss_2d[i][j]:
                        complete_class_data = data[(data[:,-1] == data[i][-1]) & (hasmiss_2d[:,j] == False)]
                        cnt_dict = defaultdict(lambda: 0)
                        for rowitem in complete_class_data:
                            cnt_dict[rowitem[j]] += 1
                        max_cnt = 0
                        fill_value = None
                        for key in cnt_dict.keys():
                            if cnt_dict[key] > max_cnt:
                                fill_value = key
                                max_cnt = cnt_dict[key]
                        data[i, j] = fill_value
    data = data.astype(np.float32)
    return data[:,:-1], data[:,-1]
                        

