max_len = 0
label_sum = [0, 0]
with open("data/train.txt", "r", encoding="utf-8") as f:
    train_lines = f.readlines()
for line in train_lines:
    tmp = line.strip().split('\t')
    sentence, label = tmp[1], tmp[2]
    label_sum[int(label)] += 1
    if len(sentence) > max_len:
        max_len = len(sentence)

print(max_len)
print(label_sum[0])
print(label_sum[1])
print(label_sum[0] + label_sum[1])
print(label_sum[0] / label_sum[1])

max_len = 0
with open("data/test_v3.txt", "r", encoding="utf-8") as f:
    train_lines = f.readlines()
for line in train_lines:
    tmp = line.strip().split('\t')
    sentence = tmp[1]
    if len(sentence) > max_len:
        max_len = len(sentence)

print(max_len)
print(len(train_lines))