import os
import time

from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import configparser

from preprocess import load_arff, fill_miss


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')
    with open(os.path.join(config['path']['OUTPUT_PATH'], 'bagging_acc.md'), 'w') as of:
        of.write('||naive bayes|decision tree|KNN|MLP|LinearSVM|\n|:---:|:---:|:---:|:---:|:---:|:---:|\n')
        for i in range(10):
            fpath = config['path']['df' + f'{i}']

            data = load_arff(fpath)
            X, Y = fill_miss(data)
            of.write(f"|{fpath[fpath.rfind('/')+1:fpath.rfind('.')]}|")
            for i in range(5):
                print(fpath[fpath.rfind('/') + 1:fpath.rfind('.')] + '\t' + config['alg']['alg' + f'{i}'], end='\t')
                start_time = time.time()
                if i == 0:
                    clf = BaggingClassifier(base_estimator=GaussianNB(),
                                            n_estimators=10,
                                            max_samples=0.5,
                                            max_features=0.5)
                elif i == 1:
                    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0, criterion='gini'),
                                            n_estimators=10,
                                            max_samples=0.5,
                                            max_features=0.5)
                elif i == 2:
                    clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),
                                            n_estimators=10,
                                            max_samples=0.5,
                                            max_features=0.5)
                elif i == 3:
                    clf = BaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(100),
                                            activation='relu', solver='adam', batch_size=128,
                                            alpha=1e-4, learning_rate_init=1e-3, learning_rate='adaptive',
                                            tol=1e-4, max_iter=200),
                                            n_estimators=10,
                                            max_samples=0.5,
                                            max_features=0.5)
                elif i == 4:
                    clf = BaggingClassifier(base_estimator=LinearSVC(penalty='l2', random_state=0, tol=1e-4),
                                            n_estimators=10,
                                            max_samples=0.5,
                                            max_features=0.5)
                accuracy = cross_val_score(clf, X, Y, cv=10).mean()
                of.write(f'{accuracy:.6f}|')
                print(f'{time.time() - start_time:.3f}s')
            of.write('\n')
