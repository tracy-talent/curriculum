import codecs

count = 0
w = codecs.open('MF1833051.txt', 'w', encoding='utf-8')
line_ids = []
predictions = ['0'] * 19990
for i, f in enumerate(['1450', '1616', '1766', '1916', '2050']):
    for index, line in enumerate(open('test.predict-'+f)):
        segs = line.strip('\n').split('\t')
        if i == 0:
            line_ids.append(segs[0])
        if segs[1] == '1':
            predictions[index] = '1'
for i, label in zip(line_ids, predictions):
    if label == '1':
        count += 1
    w.write(i+'\t'+label+'\n')
w.close()
print(count)