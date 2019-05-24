import os
import matplotlib.pyplot as plt
import numpy as np
import configparser
from matplotlib.backends.backend_pdf import PdfPages

config = configparser.ConfigParser()
config.read('../config.ini')

if not os.path.exists(os.path.join(config['path']['IMAGES'], 'single_bagging_acc')):
    os.makedirs(os.path.join(config['path']['IMAGES'], 'single_bagging_acc'))
pdf = PdfPages(os.path.join(config['path']['IMAGES'], 'single_bagging_acc/algs_single_bagging.pdf'))
data = []
bagging_data = []
with open('../output/naive_bayes_gaussian.md', 'r') as ff:
    data.append([])
    ff.readline()
    ff.readline()
    for line in ff:
        data[-1].append(line.split('|')[2])
with open('../output/decision_tree_gini.md', 'r') as ff:
    data.append([])
    ff.readline()
    ff.readline()
    for line in ff:
        data[-1].append(line.split('|')[2])
with open('../output/knn.md', 'r') as ff:
    data.append([])
    ff.readline()
    ff.readline()
    for line in ff:
        data[-1].append(line.split('|')[2])
with open('../output/mlp_100h.md', 'r') as ff:
    data.append([])
    ff.readline()
    ff.readline()
    for line in ff:
        data[-1].append(line.split('|')[2])
with open('../output/linearsvm.md', 'r') as ff:
    data.append([])
    ff.readline()
    ff.readline()
    for line in ff:
        data[-1].append(line.split('|')[2])
with open('../output/bagging_acc.md', 'r') as ff:
    ff.readline()
    ff.readline()
    for line in ff:
        bagging_data.append(line.split('|')[2:-1])
data = np.array(data).astype(np.float32)
bagging_data = np.array(bagging_data).astype(np.float32)
print('data.shape', data.shape)
print('bagging_data.shape', bagging_data.shape)

for i in range(10):
    fpath = config['path']['df' + f'{i}']
    men_means = data[:, i]
    women_means = bagging_data[i, :]
    ind = np.arange(len(men_means))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, men_means, width,
                    label='single')
    rects2 = ax.bar(ind + width/2, women_means, width,
                    label='bagging')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('accuracy')
    ax.set_title(f"classifiers on {fpath[fpath.rfind('/')+1:fpath.rfind('.')]} dataset")
    ax.set_xticks(ind)
    ax.set_xticklabels(('naive bayes', 'decision tree', 'knn', 'mlp', 'linear svm'))
    plt.yticks(np.arange(0, 1.4, 0.1))
    ax.legend()


    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{0:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos]*3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom', fontsize=9)


    autolabel(rects1, "center")
    autolabel(rects2, "center")

    fig.tight_layout()
    plt.savefig(f"../imgs/single_bagging_acc/{fpath[fpath.rfind('/')+1:fpath.rfind('.')]}.png")
    pdf.savefig(fig)

pdf.close()
# plt.show()