import codecs
import random

positive_content = []
negative_content = []
for line in codecs.open('train.txt', 'r', encoding='utf-8'):
    segs = line.strip('\n').split('\t')
    text = segs[1]
    label = segs[2]
    if label == '1':
        positive_content.append([text, label])
    else:
        negative_content.append([text, label])
random.shuffle(positive_content)
random.shuffle(negative_content)

dev_content = positive_content[:1000] + negative_content[:1000]
random.shuffle(dev_content)
with codecs.open('smooth.dev', 'w', encoding='utf-8') as dw:
    for item in dev_content:
        dw.write(item[0]+'\t'+item[1]+'\n')

train_content = positive_content[1000:] + negative_content[1000:]
random.shuffle(train_content)
with codecs.open('smooth.train', 'w', encoding='utf-8') as tw:
    for item in train_content:
        tw.write(item[0]+'\t'+item[1]+'\n')
