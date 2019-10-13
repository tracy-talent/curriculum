trnfile = 'E:\高考地理\Bert\output3\\recode.txt'
with open(trnfile, 'r', encoding='utf-8') as f:
    docs = f.readlines()
index = 1000001
t = 0

with open('E:\高考地理\Bert\output3\MG1833031.txt', 'w', encoding='utf-8') as fw:
    fw.write('1000000' + '\t' + '1' + '\n')
    for line in docs:

        if not line.__contains__('['):
            t+=1
            pass
        else:
            line = line.replace('[', '').replace(']', '').replace('\n', '')
            # print('line:'+line)
            if line.__contains__('\t'):
                sp = line.split('\t')
            else:
                sp = line.split(' ')
            # print('sp:'+sp[0])
            p1 = float(sp[0])
            a = '1'
            if p1 > 0.973:
                a = '0'
            out = str(index) + '\t' + a + '\n'
            # print(out)
            # print(line)
            fw.write(out)
            index = index + 1
print(t)
print(index-10001)


