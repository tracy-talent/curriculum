import os
import time
import argparse
import numpy as np
import json
import torch
import torch.backends.cudnn as cudnn
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

class Preprocess(object):

    def __init__(self, args, tokenizer, bert_model):
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.input_train = args.input_train
        self.input_test = args.input_test
        self.output_dir = args.output_dir
        self.use_gpu = args.use_gpu
        self.load_data()

    def load_data(self):
        with open(self.input_train, "r", encoding="utf-8") as f:
            self.train_lines = f.readlines()
        with open(self.input_test, "r", encoding="utf-8") as f:
            self.test_lines = f.readlines()

    def deal_sentence(self, sentence):
        tokens_t = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]']
        for item in tokens_t:
            tokens.append(item)
        tokens.append('[SEP]')

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_type = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        input_ids = torch.unsqueeze(torch.LongTensor(np.array(input_ids)), 0)
        tokens_type = torch.unsqueeze(torch.LongTensor(np.array(tokens_type)), 0)
        input_mask = torch.unsqueeze(torch.LongTensor(np.array(input_mask)), 0)

        if self.use_gpu:
            input_ids, tokens_type, input_mask = input_ids.cuda(), tokens_type.cuda(), input_mask.cuda()

        data, _ = self.bert_model(input_ids, tokens_type, input_mask, False)

        data = data[0].detach().data.cpu().numpy()
        # print(data.shape)
        return data.tolist()


    def preprocess_train(self):
        self.data_dict = {}
        start = time.time()
        label_sum = [0, 0]
        for line in self.train_lines:
            tmp = line.strip().split("\t")
            sentence_id = tmp[0]
            input_ids = self.deal_sentence(tmp[1])
            sentence_label = tmp[2]
            self.data_dict[sentence_id] = \
                {"embedding": input_ids, "label": sentence_label}
            label_sum[int(sentence_label)] += 1
            print("sentence_id: {} | time: {:.2f}s".format(sentence_id, time.time() - start))
            # break

        print("positive num: {} | nagetive num: {}".format(label_sum[0], label_sum[1]))

    def preprocess_test(self):
        self.test_dict = {}
        start = time.time()
        for line in self.test_lines:
            tmp = line.strip()
            sentence_id = tmp[0:7]
            input_ids = self.deal_sentence(tmp[8:])
            self.test_dict[sentence_id] = \
                {"embedding": input_ids}
            print("sentence_id: {} | time: {:.2f}s".format(sentence_id, time.time() - start))
            # break

    def shuffle(self):
        keys = np.array(list(self.data_dict.keys()))
        np.random.shuffle(keys)
        size = len(keys)
        train_keys = keys[:int(0.75 * size)]
        dev_keys = keys[int(0.75 * size):]
        self.train_dict = {}
        self.dev_dict = {}
        for key in train_keys:
            self.train_dict[key] = self.data_dict[key]
        for key in dev_keys:
            self.dev_dict[key] = self.data_dict[key]
        print("train data num: {} | dev data num: {}".format(len(self.train_dict), len(self.dev_dict)))
        assert len(self.train_dict) + len(self.dev_dict) == len(self.data_dict)

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

parser.add_argument('--input-train', type=str, default="data/train.txt")
parser.add_argument('--input-test', type=str, default="data/test.content.txt")
parser.add_argument('--output-dir', type=str, default="preprocessed_embedding")
parser.add_argument('--gpu-devices', type=str, default="0")
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    args.use_gpu = use_gpu

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    bert_model = BertModel.from_pretrained("bert-base-chinese")

    if use_gpu:
        bert_model = bert_model.cuda()

    processor = Preprocess(args, tokenizer, bert_model)
    processor.do_preprocess()


if __name__ == "__main__":
    main()
