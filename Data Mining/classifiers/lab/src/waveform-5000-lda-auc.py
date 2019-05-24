import os
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import configparser

from preprocess import load_arff, fill_miss


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')
    with open(os.path.join(config['path']['OUTPUT_PATH'], 'waveform-5000-lda-auc.md'), 'w') as of:
        of.write('* waveform-5000 dataset\n\n')
        of.write('| |source|LDA|\n|:---:|:---:|:---:|\n')
        start_time = time.time()
        fpath = config['path']['df' + f'{9}']
        data = load_arff(fpath)
        X, y = fill_miss(data)
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_new = lda.fit_transform(X, y)
        fig = plt.figure()
        plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
        plt.savefig('../imgs/waveform-lda.png')
        n_classes = np.arange(np.unique(y).size)
        skf = StratifiedKFold(n_splits=10)
        for i in range(5):
            print(config['alg']['alg'+f'{i}'], end='\t')
            if i == 0:
                clf = GaussianNB()
            elif i == 1:
                clf = DecisionTreeClassifier(random_state=0, criterion='gini')
            elif i == 2:
                clf = KNeighborsClassifier(n_neighbors=3)
            elif i == 3:
                clf = MLPClassifier(hidden_layer_sizes=(100),
                                    activation='relu',
                                    solver='adam',
                                    batch_size=128,
                                    alpha=1e-4,
                                    learning_rate_init=1e-3,
                                    learning_rate='adaptive',
                                    tol=1e-4,
                                    max_iter=200)
            elif i == 4:
                clf = LinearSVC(penalty='l2', random_state=0, tol=1e-4)
            skf_accuracy1 = []
            skf_accuracy2 = []
            for train, test in skf.split(X, y):
                clf.fit(X[train], y[train])
                if n_classes.size < 3:
                    skf_accuracy1.append(roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1] if i != 4 else clf.decision_function(X[test]), average='micro'))
                    clf.fit(X_new[train], y[train])
                    skf_accuracy2.append(roc_auc_score(y[test], clf.predict_proba(X_new[test])[:, 1] if i != 4 else clf.decision_function(X_new[test]), average='micro'))
                else:
                    ytest_one_hot = label_binarize(y[test], n_classes)
                    skf_accuracy1.append(roc_auc_score(ytest_one_hot, clf.predict_proba(X[test]) if i != 4 else clf.decision_function(X[test]), average='micro'))
                    clf.fit(X_new[train], y[train])
                    skf_accuracy2.append(roc_auc_score(ytest_one_hot, clf.predict_proba(X_new[test]) if i != 4 else clf.decision_function(X_new[test]), average='micro'))
            accuracy1 = np.mean(skf_accuracy1)
            accuracy2 = np.mean(skf_accuracy2)
            of.write(f"|{config['alg']['alg'+f'{i}']}|{accuracy1:.6f}|{accuracy2:.6f}|\n")
            print(f'{time.time() - start_time:.3f}s')
