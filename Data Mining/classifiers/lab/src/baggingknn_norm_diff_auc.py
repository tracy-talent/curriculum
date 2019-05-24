import os
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import configparser

from preprocess import load_arff, fill_miss


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')
    with open(os.path.join(config['path']['OUTPUT_PATH'], 'bagging_knn_isnorm_auc.md'), 'w') as of:
        of.write('| |source|norm|\n|:---:|:---:|:---:|\n')
        for i in range(10):
            start_time = time.time()
            fpath = config['path']['df' + f'{i}']
            print(fpath[fpath.rfind('/')+1:fpath.rfind('.')], end='\t')
            data = load_arff(fpath)
            X, y = fill_miss(data)
            n_classes = np.arange(np.unique(y).size)
            clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),
                                    n_estimators=10,
                                    max_samples=0.5,
                                    max_features=0.5)
            skf = StratifiedKFold(n_splits=10)
            skf_accuracy = []
            skf_accuracy_norm = []
            for train, test in skf.split(X, y):
                clf.fit(X[train], y[train])
                scaler = StandardScaler().fit(X[train])
                if n_classes.size < 3:
                    skf_accuracy.append(roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1], average='micro'))
                    clf.fit(scaler.transform(X[train]), y[train])
                    skf_accuracy_norm.append(roc_auc_score(y[test], clf.predict_proba(scaler.transform(X[test]))[:, 1], average='micro'))
                else:
                    ytest_one_hot = label_binarize(y[test], n_classes)
                    skf_accuracy.append(roc_auc_score(ytest_one_hot, clf.predict_proba(X[test]), average='micro'))
                    clf.fit(scaler.transform(X[train]), y[train])
                    skf_accuracy_norm.append(roc_auc_score(ytest_one_hot, clf.predict_proba(scaler.transform(X[test])), average='micro'))
            accuracy1 = np.mean(skf_accuracy)
            accuracy2 = np.mean(skf_accuracy_norm)
            of.write(f"|{fpath[fpath.rfind('/')+1:fpath.rfind('.')]}|{accuracy1:.6f}|{accuracy2:.6f}|\n")
            print(f'{time.time() - start_time:.3f}s')
