import os
import time
import argparse
import numpy as np
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer

from pytorch_pretrained_bert.modeling import BertModel

class Preprocess(object):

    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.input_train = args.input_train
        self.input_test = args.input_test
        self.output_dir = args.output_dir
        self.load_data()

    def load_data(self):
        with open(self.input_train, "r", encoding="utf-8") as f:
            self.train_lines = f.readlines()
        with open(self.input_test, "r", encoding="utf-8") as f:
            self.test_lines = f.readlines()
        with open("data/test.content.real.txt", "r", encoding="utf-8") as f:
            extra_data = f.readlines()
        with open("data/test.label.real.txt", "r", encoding="utf-8") as f:
            extra_label = f.readlines()
        assert len(extra_data) == len(extra_label)
        for index in range(len(extra_data)):
            data_split = extra_data[index].strip().split("\t")
            label_split = extra_label[index].strip().split("\t")
            line = "{}\t{}\t{}".format(data_split[0], data_split[1], label_split[1])
            self.train_lines.append(line)

    def deal_sentence(self, sentence):
        tokens_t = self.tokenizer.tokenize(sentence)
        # tokens = ['[CLS]']
        tokens = []
        for item in tokens_t:
            tokens.append(item)
        # tokens.append('[SEP]')
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return input_ids

    def preprocess_train(self):
        self.p_data_dict = {}
        self.n_data_dict = {}
        start = time.time()
        label_sum = [0, 0]
        for line in self.train_lines:
            tmp = line.strip().split("\t")
            sentence_id = tmp[0]
            input_ids = self.deal_sentence(tmp[1])
            sentence_label = tmp[2]
            assert sentence_label == "0" or sentence_label == "1"
            if sentence_label == "0":
                self.p_data_dict[sentence_id] = \
                    {"ids": input_ids, "label": sentence_label}
            else:
                self.n_data_dict[sentence_id] = \
                    {"ids": input_ids, "label": sentence_label}
            label_sum[int(sentence_label)] += 1
            # print("sentence_id: {} | time: {:.2f}s".format(sentence_id, time.time() - start))

        print("positive num: {} | nagetive num: {}".format(label_sum[0], label_sum[1]))

    def preprocess_test(self):
        self.test_dict = {}
        start = time.time()
        for line in self.test_lines:
            tmp = line.strip()
            sentence_id = tmp[0:7]
            input_ids = self.deal_sentence(tmp[8:])
            self.test_dict[sentence_id] = \
                {"ids": input_ids}
            # print("sentence_id: {} | time: {:.2f}s".format(sentence_id, time.time() - start))

    def shuffle(self):
        p_keys = np.array(list(self.p_data_dict.keys()))
        np.random.shuffle(p_keys)
        n_keys = np.array(list(self.n_data_dict.keys()))
        np.random.shuffle(n_keys)

        print("positive key: {} | nagetive key: {}".format(len(p_keys), len(n_keys)))

        dev_num = 200

        print("rate: {}".format(float(len(p_keys) - dev_num) / (len(n_keys) - dev_num)))

        p_train_keys = p_keys[dev_num:]
        n_train_keys = n_keys[dev_num:]
        p_dev_keys = p_keys[:dev_num]
        n_dev_keys = n_keys[:dev_num]

        self.train_dict = {}
        self.dev_dict = {}
        for key in p_train_keys:
            self.train_dict[key] = self.p_data_dict[key]
        for key in n_train_keys:
            self.train_dict[key] = self.n_data_dict[key]
        for key in p_dev_keys:
            self.dev_dict[key] = self.p_data_dict[key]
        for key in n_dev_keys:
            self.dev_dict[key] = self.n_data_dict[key]

        print("train data num: {} | dev data num: {}".format(len(self.train_dict), len(self.dev_dict)))
        # assert len(self.train_dict) + len(self.dev_dict) == len(self.data_dict)

    def write_down(self):
        with open(os.path.join(self.output_dir, "test.json"), "w") as f:
            json.dump(self.test_dict, f)
        with open(os.path.join(self.output_dir, "train.json"), "w") as f:
            json.dump(self.train_dict, f)
        with open(os.path.join(self.output_dir, "dev.json"), "w") as f:
            json.dump(self.dev_dict, f)

    def do_preprocess(self):
        self.preprocess_train()
        self.preprocess_test()
        self.shuffle()
        self.write_down()


parser = argparse.ArgumentParser(description="preprocess data with bert embedding.")

parser.add_argument('--input-train', type=str, default="data/train_downsample.txt")
parser.add_argument('--input-test', type=str, default="data/test_v3.txt")
parser.add_argument('--output-dir', type=str, default="preprocessed_data")

args = parser.parse_args()


def main():

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    processor = Preprocess(args, tokenizer)
    processor.do_preprocess()

    # with open("/Users/limingwei/Desktop/MG1833039.txt", "r") as f:
    #     lines = f.readlines()
    # label = [0, 0]
    # for line in lines:
    #     l = line.strip().split("\t")
    #     label[int(l[1])] += 1
    # print(label)

if __name__ == "__main__":
    main()
