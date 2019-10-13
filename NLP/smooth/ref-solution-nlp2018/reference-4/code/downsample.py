import os
import time
import argparse
import numpy as np


class DownSample(object):

    def __init__(self, input_file, output_file, weight):
        self.input_file = input_file
        self.output_file = output_file
        self.weight = weight

        with open(self.input_file, "r", encoding="utf-8") as f:
            self.lines = f.readlines()

    def do_downsample(self):
        positive_dict = {}
        nagetive_dict = {}
        for line in self.lines:
            tmp = line.strip().split("\t")
            id = tmp[0]
            sentence = tmp[1]
            label = tmp[2]
            assert label == "0" or label == "1"
            if label == "0":
                positive_dict[id] = {"sentence": sentence, "label": label}
            else:
                nagetive_dict[id] = {"sentence": sentence, "label": label}

        keys = list(positive_dict.keys())
        np.random.shuffle(keys)
        sample_size = len(nagetive_dict)
        sample_keys = keys[:int(0.5 * sample_size)]

        sample_positive_dict = {}
        for key in sample_keys:
            sample_positive_dict[key] = positive_dict[key]

        with open(self.output_file, "w", encoding="utf-8") as f:
            for key, value in sample_positive_dict.items():
                f.write("{}\t{}\t{}\n".format(key, value["sentence"], value["label"]))
            for key, value in nagetive_dict.items():
                f.write("{}\t{}\t{}\n".format(key, value["sentence"], value["label"]))

def main():
    downsampler = DownSample(input_file="data/train.txt",
                             output_file="data/train_downsample.txt",
                             weight=1.0)
    downsampler.do_downsample()


if __name__ == "__main__":
    main()