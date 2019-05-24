import os
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import numpy as np
import configparser

from preprocess import load_arff, fill_miss


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')
    with open(os.path.join(config['path']['OUTPUT_PATH'], 'naive_bayes_gaussian.md'), 'w') as of:
        of.write('||accuracy|auc|\n|:---:|:---:|:---:|\n')
        for i in range(10):
            start_time = time.time()
            fpath = config['path']['df' + f'{i}']
            print(fpath[fpath.rfind('/')+1:fpath.rfind('.')], end='\t')
            data = load_arff(fpath)
            X, y = fill_miss(data)
            data = data.astype(np.float32)
            clf = GaussianNB()
            skf = StratifiedKFold(n_splits=10)
            skf_accuracy1 = []
            skf_accuracy2 = []
            n_classes = np.arange(np.unique(y).size)
            for train, test in skf.split(X, y):
                clf.fit(X[train], y[train])
                skf_accuracy1.append(clf.score(X[test], y[test]))
                if n_classes.size < 3:
                    skf_accuracy2.append(roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1], average='micro'))
                else:
                    ytest_one_hot = label_binarize(y[test], n_classes)
                    skf_accuracy2.append(roc_auc_score(ytest_one_hot, clf.predict_proba(X[test]), average='micro'))
            accuracy1 = np.mean(skf_accuracy1)
            accuracy2 = np.mean(skf_accuracy2)
            of.write(f"|{fpath[fpath.rfind('/')+1:fpath.rfind('.')]}|{accuracy1:.6f}|{accuracy2:.6f}|\n")
            print(f'{time.time() - start_time:.3f}s')
