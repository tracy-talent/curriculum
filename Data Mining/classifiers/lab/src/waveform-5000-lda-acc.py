import os
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import configparser

from preprocess import load_arff, fill_miss


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')
    with open(os.path.join(config['path']['OUTPUT_PATH'], 'waveform-5000-lda-acc.md'), 'w') as of:
        of.write('| |source|LDA|\n|:---:|:---:|:---:|\n')
        start_time = time.time()
        fpath = config['path']['df' + f'{9}']
        data = load_arff(fpath)
        X, Y = fill_miss(data)
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_new = lda.fit_transform(X, Y)
        fig = plt.figure()
        plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=Y)
        if not os.path.exists(os.path.join(config['path']['IMAGES'], 'lda_waveform')):
            os.makedirs(os.path.join(config['path']['IMAGES'], 'lda_waveform'))
        plt.savefig('../imgs/lda_waveform/lda_effect.png')
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
            accuracy1 = cross_val_score(clf, X, Y, cv=10).mean()
            accuracy2 = cross_val_score(clf, X_new, Y, cv=10).mean()
            of.write(f"|{config['alg']['alg'+f'{i}']}|{accuracy1:.6f}|{accuracy2:.6f}|\n")
            print(f'{time.time() - start_time:.3f}s')
