import os
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
import configparser

from preprocess import load_arff, fill_miss


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')
    with open(os.path.join(config['path']['OUTPUT_PATH'], 'knn_isminmax.md'), 'w') as of:
        of.write('| |source|minmax|\n|:---:|:---:|:---:|\n')
        for i in range(10):
            start_time = time.time()
            fpath = config['path']['df' + f'{i}']
            print(fpath[fpath.rfind('/')+1:fpath.rfind('.')], end='\t')
            data = load_arff(fpath)
            X, y = fill_miss(data)
            clf = KNeighborsClassifier(n_neighbors=3)
            kfold_accuracy = []
            kfold_accuracy_norm = []
            kf = StratifiedKFold(n_splits=10)
            for train, test in kf.split(X, y):
                clf.fit(X[train], y[train])
                kfold_accuracy.append(clf.score(X[test], y[test]))
                scaler = MinMaxScaler().fit(X[train])
                clf.fit(scaler.transform(X[train]), y[train])
                kfold_accuracy_norm.append(clf.score(scaler.transform(X[test]), y[test]))
            accuracy1 = np.mean(kfold_accuracy)
            accuracy2 = np.mean(kfold_accuracy_norm)
            of.write(f"|{fpath[fpath.rfind('/')+1:fpath.rfind('.')]}|{accuracy1:.6f}|{accuracy2:.6f}|\n")
            print(f'{time.time() - start_time:.3f}s')
