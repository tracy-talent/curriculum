import random
import functools
import numpy as np
import tensorflow as tf

#################################
####### generate function #######
#################################

def generate_fn(*args):
    for sample in zip(*args):
        yield sample


#########################################
####### DataLoader 1: for LSTM #######
#########################################

class DataLoader_LSTM(object):
    def __init__(self, 
                 input_seq_path, 
                 output_seq_path, 
                 w2i_char,
                 w2i_bio):
        
        inputs_seq = []
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in line.split(" ")]
                inputs_seq.append(seq)
                
        outputs_seq = []
        with open(output_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_bio[word] for word in line.split(" ")]
                outputs_seq.append(seq)
                    
        assert len(inputs_seq) == len(outputs_seq)
        assert all(len(input_seq) == len(output_seq) for input_seq, output_seq in zip(inputs_seq, outputs_seq))

        # padding
        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.inputs_seq = inputs_seq
        self.inputs_seq_len = [len(seq) for seq in inputs_seq]
        self.outputs_seq = outputs_seq
        print("DataProcessor load data num: " + str(len(inputs_seq)))


    def get_dataset(self, batch_size=16, epoches=5, buffer_size=1):
        shapes = ([None], (), [None])
        types = (tf.int32, tf.int32, tf.int32)
        defaults = (self.w2i_char['[PAD]'], 0, self.w2i_bio['O'])
        dataset = tf.data.Dataset.from_generator(
            functools.partial(generate_fn, self.inputs_seq, self.inputs_seq_len, self.outputs_seq),
            output_shapes=shapes, output_types=types)
        dataset = dataset.padded_batch(batch_size, padded_shapes=shapes, padding_values=defaults, drop_remainder=False) \
                .repeat(epoches) \
                .shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True) \
                .prefetch(1)

        return dataset

##########################
####### Vocabulary #######
##########################
            
def load_vocabulary(path):
    vocab = open(path, "r", encoding="utf-8").read().strip().split("\n")
    print("load vocab from: {}, containing words: {}".format(path, len(vocab)))
    w2i = {}
    i2w = {}
    for i, w in enumerate(vocab):
        w2i[w] = i
        i2w[i] = w
    return w2i, i2w

######################################
####### extract_kvpairs_by_bio #######
######################################

def extract_kvpairs_in_bio(bio_seq, word_seq):
    assert len(bio_seq) == len(word_seq)
    pairs = set()
    pre_bio = "O"
    v = ""
    for i, bio in enumerate(bio_seq):
        if (bio == "O"):
            if v != "": pairs.add((pre_bio[2:], v))
            v = ""
        elif (bio[0] == "B"):
            if v != "": pairs.add((pre_bio[2:], v))
            v = word_seq[i]
        elif (bio[0] == "I"):
            if (pre_bio[0] == "O") or (pre_bio[2:] != bio[2:]):
                if v != "": pairs.add((pre_bio[2:], v))
                v = ""
            else:
                v += word_seq[i]
        pre_bio = bio
    if v != "": pairs.add((pre_bio[2:], v))
    return pairs

def extract_kvpairs_in_bioes(bio_seq, word_seq, attr_seq):
    assert len(bio_seq) == len(word_seq) == len(attr_seq)
    pairs = set()
    v = ""
    for i in range(len(bio_seq)):
        word = word_seq[i]
        bio = bio_seq[i]
        attr = attr_seq[i]
        if bio == "O":
            v = ""
        elif bio == "S":
            v = word
            pairs.add((attr, v))
            v = ""
        elif bio == "B":
            v = word
        elif bio == "I":
            if v != "": 
                v += word
        elif bio == "E":
            if v != "":
                v += word
                pairs.add((attr, v))
            v = ""
    return pairs


############################
####### cal_f1_score #######
############################

def cal_f1_score(preds, golds):
    assert len(preds) == len(golds)
    p_sum = 0
    r_sum = 0
    hits = 0
    for pred, gold in zip(preds, golds):
        p_sum += len(pred)
        r_sum += len(gold)
        for label in pred:
            if label in gold:
                hits += 1
    p = hits / p_sum if p_sum > 0 else 0
    r = hits / r_sum if r_sum > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

