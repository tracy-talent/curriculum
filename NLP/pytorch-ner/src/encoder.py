"""
 Author: liujian
 Date: 2021-04-21 21:14:53
 Last Modified by: liujian
 Last Modified time: 2021-04-21 21:14:53
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):

    def __init__(self, 
                 token2id, 
                 max_length=256, 
                 embedding_size=50,
                 blank_padding=True):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            embedding_size: size of word embedding
            blank_padding: padding for CNN
        """
        # Hyperparameters
        super().__init__()

        self.token2id = token2id
        self.id2token = {tid:t for t, tid in token2id.items()}
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.blank_padding = blank_padding

        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
        self.word_embedding = nn.Embedding(len(token2id), self.embedding_size)

    def forward(self, seqs, att_mask):
        """
        Args:
            seqs: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, L, H), representations for sequences
        """
        # Check size of tensors
        inputs_embed = self.word_embedding(seqs) # (B, L, EMBED)
        return inputs_embed
    
    def tokenize(self, *items):
        """
        Args:
            items: (tokens, tags) or (tokens, spans, atrrs) or (sentence)
        Return:
            index number of tokens and positions             
        """
        tokens = items[0]

        avail_len = torch.tensor([len(tokens)]) # 序列实际长度

        # Token -> index
        indexed_tokens = [self.token2id[token] if token in self.token2id else self.token2id['[UNK]'] for token in tokens]
        if self.blank_padding:
            indexed_tokens = indexed_tokens[:self.max_length]
            indexed_tokens.extend([self.token2id['[PAD]']] * (self.max_length - len(indexed_tokens)))
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)

        # attention mask
        att_mask = torch.zeros(indexed_tokens.size(), dtype=torch.uint8) # (1, L), crf reqiure mask dtype=bool or uint8
        att_mask[0, :avail_len] = 1

        return indexed_tokens, att_mask
