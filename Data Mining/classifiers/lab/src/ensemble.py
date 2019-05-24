import os
import time

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import configparser

from preprocess import load_arff, fill_miss


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')
    with open(os.path.join(config['path']['OUTPUT_PATH'], 'extratrees.md'), 'w') as of:
        of.write('||cross_val_score|KFold|\n|:---:|:---:|:---:|\n')
        for i in range(10):
            start_time = time.time()
            fpath = config['path']['df' + f'{i}']
            print(fpath[fpath.rfind('/')+1:fpath.rfind('.')], end='\t')
            data = load_arff(fpath)
            X, Y = fill_miss(data)
            data = data.astype(np.float32)
            # clf = RandomForestClassifier(n_estimators=10)
            clf = ExtraTreesClassifier(n_estimators=10,
                                       max_depth=None,
                                       min_samples_split=2,
                                       random_state=0)
            # clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), 
            #                         n_estimators=10,
            #                         max_samples=0.5,
            #                         max_features=0.5)
            accuracy1 = cross_val_score(clf, X, Y, cv=10).mean()
            kfold_accuracy = []
            kf = KFold(n_splits=10)
            for train, test in kf.split(data):
                clf.fit(X[train], Y[train])
                kfold_accuracy.append(clf.score(X[test], Y[test]))
            accuracy2 = np.mean(kfold_accuracy)
            of.write(f"|{fpath[fpath.rfind('/')+1:fpath.rfind('.')]}|{accuracy1:.6f}|{accuracy2:.6f}|\n")
            print(f'{time.time() - start_time:.3f}s')
