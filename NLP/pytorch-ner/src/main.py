"""
 Author: liujian
 Date: 2021-04-21 20:57:56
 Last Modified by: liujian
 Last Modified time: 2021-04-21 20:57:56
"""

# coding:utf-8
import sys
from utils import get_logger, fix_seed, load_vocab
from trainer import NER_Trainer
from encoder import BaseEncoder
from model import BILSTM_CRF

import torch
import numpy as np
import json
import os
import re
import datetime
import argparse
import configparser


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--use_lstm', action='store_true', 
        help='whether add lstm encoder on top of bert')
parser.add_argument('--use_crf', action='store_true', 
        help='whether use crf for sequence decode')
parser.add_argument('--tagscheme', default='bio', type=str,
        help='the sequence tag scheme')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'micro_p', 'micro_r'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['renmin1998'], 
        help='Dataset')
parser.add_argument('--compress_seq', action='store_true', 
        help='whether use pack_padded_sequence to compress mask tokens of batch sequence, off when parallel')

# Hyper-parameters
parser.add_argument('--embedding_size', default=100, type=int,
        help='embedding size')
parser.add_argument('--hidden_size', default=100, type=int,
        help='hidden size')
parser.add_argument('--batch_size', default=64, type=int,
        help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float,
        help='Learning rate')
parser.add_argument('--dropout_rate', default=0.1, type=float,
        help='dropout rate')
parser.add_argument('--optimizer', default='adam', type=str, choices=['sgd', 'adam', 'adamw'],
        help='optimizer')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=3, type=int,
        help='Max number of training epochs')
parser.add_argument('--warmup_epoches', default=0, type=int,
        help='warmup epoches for learning rate scheduler')
parser.add_argument('--random_seed', default=12345, type=int,
                    help='global random seed')

args = parser.parse_args()

project_path = '/'.join(os.path.abspath(__file__).split('/')[:-3])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))

# set global random seed
fix_seed(args.random_seed)

# construct save path name
def make_dataset_name():
    dataset_name = args.dataset + '_' + args.tagscheme
    return dataset_name
def make_model_name():
    model_name = ''
    if args.use_lstm:
        model_name += '_bilstm' if model_name != '' else 'bilstm'
    if args.use_crf:
        model_name += '_crf' if model_name != '' else 'crf'
    return model_name
dataset_name = make_dataset_name()
model_name = make_model_name()

# logger
os.makedirs(os.path.join('../output', dataset_name, model_name), exist_ok=True)
logger = get_logger(sys.argv, os.path.join('../output', dataset_name, model_name, 
                        f'{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}.log')) 

# Some basic settings
os.makedirs(os.path.join('../output', dataset_name, 'ckpt'), exist_ok=True)
if len(args.ckpt) == 0:
    args.ckpt = model_name
ckpt = os.path.join('../output', dataset_name, 'ckpt', f'{args.ckpt}_0.pth.tar')
ckpt_cnt = 0
while os.path.exists(ckpt):
    ckpt_cnt += 1
    ckpt = re.sub('\d+\.pth\.tar', f'{ckpt_cnt}.pth.tar', ckpt)

if args.dataset != 'none':
    args.train_file = os.path.join('../input', args.dataset, f'train.char.{args.tagscheme}')
    args.val_file = os.path.join('../input', args.dataset, f'val.char.{args.tagscheme}')
    args.test_file = os.path.join('../input', args.dataset, f'test.char.{args.tagscheme}')
    args.tag2id_file = os.path.join('../input', args.dataset, f'tag2id.{args.tagscheme}')
    args.vocab_file = os.path.join('../input', args.dataset, f'vocab_char.txt')
else:
    raise Exception('dataset is not specified')

logger.info('Arguments:')
for arg in vars(args):
    logger.info('{}: {}'.format(arg, getattr(args, arg)))

#  load tag
tag2id = load_vocab(args.tag2id_file)
token2id = load_vocab(args.vocab_file)

sequence_encoder = BaseEncoder(
    token2id=token2id,
    max_length=args.max_length,
    embedding_size=args.embedding_size,
    blank_padding=True
)

# Define the model
model = BILSTM_CRF(
    hidden_size=args.hidden_size,
    sequence_encoder=sequence_encoder, 
    tag2id=tag2id, 
    compress_seq=args.compress_seq,
    use_lstm=args.use_lstm, 
    use_crf=args.use_crf,
    dropout_rate=args.dropout_rate,
    tagscheme=args.tagscheme
)

# Define the whole training framework
trainer = NER_Trainer(
    model=model,
    train_path=args.train_file if not args.only_test else None,
    val_path=args.val_file if not args.only_test else None,
    test_path=args.test_file,
    ckpt=ckpt,
    logger=logger,
    compress_seq=args.compress_seq,
    tagscheme=args.tagscheme, 
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt=args.optimizer,
    metric=args.metric
)

# Load pretrained model
if ckpt_cnt > 0:
    logger.info('load checkpoint')
    trainer.load_model(re.sub('\d+\.pth\.tar', f'{ckpt_cnt-1}.pth.tar', ckpt))

# Train the model
if not args.only_test:
    trainer.train_model()
    trainer.load_model(ckpt)

# Test
result = trainer.eval_model(trainer.test_loader)
# Print the result
logger.info('Test set results:')
logger.info('Accuracy: {}'.format(result['acc']))
logger.info('Micro precision: {}'.format(result['micro_p']))
logger.info('Micro recall: {}'.format(result['micro_r']))
logger.info('Micro F1: {}'.format(result['micro_f1']))
logger.info('Category-P/R/F1: {}'.format(result['category-p/r/f1']))
