
with open('../input/train.data', 'r') as td:
    maxlen = 0
    cnt = 0
    for line in td:
        sent = line.strip()
        if len(sent) > maxlen:
            maxlen = len(sent)
        if len(sent) > 300:
            cnt += 1
            print(len(sent), sent)
    print(cnt, maxlen)
