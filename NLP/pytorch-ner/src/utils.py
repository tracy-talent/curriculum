"""
 Author: liujian
 Date: 2021-04-21 20:53:57
 Last Modified by: liujian
 Last Modified time: 2021-04-21 20:53:57
"""

from collections import defaultdict, OrderedDict
import logging
import os
import torch
import random
import numpy as np


def fix_seed(seed=12345): 
    """fix random seed for reproduction

    Args:
        seed (int): [description]. Defaults to 12345.
    """
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed) # gpu
    np.random.seed(seed) # numpy
    random.seed(seed) # random and transforms
    torch.backends.cudnn.deterministic=True # cudnn


def get_logger(command_argv, log_file_path=None):
    """add log
    Args:
        command_argv (list): sys.argv
        log_file_path (str): output path of log file, do not save log file when it's not specified

    Returns:
        logging.logger: logger
    """
    program = os.path.basename(command_argv[0])
    logger = logging.getLogger(program)

    # 设置输出的日志级别
    logger.setLevel(level=logging.INFO)
    # 设置日志输出内容：时间%Y-%m-%d %H:%M:%S，日志级别INFO,WARNING,ERROR，日志信息
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    if log_file_path is None:
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        logger.addHandler(chlr)

    if log_file_path:
        # 设置日志输出文件
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        fhlr = logging.FileHandler(log_file_path)
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)

    # 输出日志信息内容
    logger.info("running %s" % ' '.join(command_argv))

    return logger


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    if vocab_file ==  None:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    if isinstance(vocab_file, str):
        vocab = OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            for token in reader:
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab
    else:
        return vocab_file


def extract_kvpairs_in_bio(bio_seq, word_seq):
    assert len(bio_seq) == len(word_seq)
    pairs = list()
    pre_bio = "O"
    v = ""
    spos = -1
    for i, bio in enumerate(bio_seq):
        word = word_seq[i]
        if bio == "O":
            if v != "": 
                pairs.append(((spos, i), pre_bio[2:], v))
            v = ""
        elif bio[0] == "B":
            if v != "": 
                pairs.append(((spos, i), pre_bio[2:], v))
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bio[0] == "I":
            if pre_bio[0] == "O" or pre_bio[2:] != bio[2:] or v == "":
                if v != "": 
                    pairs.append(((spos, i), pre_bio[2:], v))
                v = ""
            else:
                v += word[2:] if word.startswith('##') else word
        pre_bio = bio
    if v != "":
        pairs.append(((spos, len(bio_seq)), pre_bio[2:], v))
    return pairs


# 严格按照BMOE一致类型抽取实体
def extract_kvpairs_in_bmoe(bioe_seq, word_seq):
    assert len(bioe_seq) == len(word_seq)
    pairs = list()
    pre_bioe = "O"
    v = ""
    spos = -1
    for i, bioe in enumerate(bioe_seq):
        word = word_seq[i]
        if bioe == "O":
            if v != "" and spos + 1 == i: 
                pairs.append(((spos, spos + 1), pre_bioe[2:], v))
            v = ""
        elif bioe[0] == "B":
            if v != "" and spos + 1 == i: 
                pairs.append(((spos, spos + 1), pre_bioe[2:], v))
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bioe[0] == "M":
            if pre_bioe[0] in "OE" or pre_bioe[2:] != bioe[2:] or v == "":
                if v != "" and spos + 1 == i: 
                    pairs.append(((spos, spos + 1), pre_bioe[2:], v))
                v = ""
            else:
                v += word[2:] if word.startswith('##') else word
        elif bioe[0] == 'E':
            if pre_bioe[0] in 'BM' and pre_bioe[2:] == bioe[2:] and v != "":
                v += word[2:] if word.startswith('##') else word
                pairs.append(((spos, i + 1), bioe[2:], v))
            v = ""
        pre_bioe = bioe
    if v != "" and spos + 1 == len(bioe_seq):
        pairs.append(((spos, spos + 1), pre_bioe[2:], v))
    return pairs

