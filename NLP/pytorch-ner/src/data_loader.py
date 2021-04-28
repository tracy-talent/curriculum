import numpy as np
import logging
from functools import partial

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def compress_sequence(seqs, lengths):
    """compress padding in batch

    Args:
        seqs (torch.LongTensor->(B, L)): batch seqs, sorted by seq's actual length in decreasing order
        lengths (torch.LongTensor->(B)): length of every seq in batch in decreasing order

    Returns:
        torch.LongTensor: compressed batch seqs
    """
    packed_seqs = pack_padded_sequence(input=seqs, lengths=lengths.detach().cpu().numpy(), batch_first=True)
    seqs, _ = pad_packed_sequence(sequence=packed_seqs, batch_first=True)
    return seqs


class NERDataset(data.Dataset):
    """
    named entity recognition dataset
    """
    def __init__(self, path, tag2id, tokenizer, preload=True, **kwargs):
        """
        Args:
            path: path of the input file
            tag2id: dictionary of entity_tag->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.preload = preload
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.kwargs = kwargs

        # Load the file
        self.corpus = [] # [[[seq_tokens], [seq_tags]], ..]
        with open(path, 'r', encoding='utf-8') as f:
            tokens = []
            for line in f.readlines():
                line = line.strip().split()
                if len(line) > 0:
                    tokens.append(line)
                elif len(tokens) > 0:
                    self.corpus.append([list(seq) for seq in zip(*tokens)])
                    tokens = []
        if self.preload:
            self._construct_data()

        self.weight = np.ones((len(self.tag2id)), dtype=np.float32)
        for item in self.corpus:
            for tag in item[1]:
                self.weight[self.tag2id[tag]] += 1.0
        self.weight = 1.0 / (self.weight ** 0.05)
        self.weight = torch.from_numpy(self.weight)

        logging.info("Loaded sentence NER dataset {} with {} lines and {} entity types.".format(path, len(self), len(self.tag2id)))

    def _getitem(self, items): # items = [[seq_tokens..], [seq_tags..]]
        seqs = list(self.tokenizer(*items, **self.kwargs))
        length = seqs[0].size(1)
        if length >= len(items[1]):
            labels = [self.tag2id[tag] for tag in items[1]]
            labels.extend([self.tag2id['O']] * (length - len(items[1])))
        else:
            labels = [self.tag2id[tag] for tag in items[1][:length]]
            labels[-1] = self.tag2id['O']
        item = [torch.tensor([labels])] + seqs # make labels size (1, L)
        return item

    def _construct_data(self):
        self.data = []
        for index in range(len(self.corpus)):
            item = self._getitem(self.corpus[index])
            self.data.append(item)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, index):
        if self.preload:
            return self.data[index]
        else:
            return self._getitem(self.corpus[index])

    @classmethod
    def collate_fn(cls, compress_seq, data):
        seqs = list(zip(*data))
        if compress_seq:
            seqs_len = torch.cat(seqs[-1], dim=0).sum(dim=-1)
            sorted_length_indices = seqs_len.argsort(descending=True) 
            seqs_len = seqs_len[sorted_length_indices]
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
                if len(seqs[i].size()) > 1 and seqs[i].size(1) > 1:
                    seqs[i] = compress_sequence(seqs[i][sorted_length_indices], seqs_len)
                else:
                    seqs[i] = seqs[i][sorted_length_indices]
        else:
            for i in range(len(seqs)):
                seqs[i] = torch.cat(seqs[i], dim=0)
        return seqs
    
    
def NERDataLoader(path, tag2id, tokenizer, batch_size, 
        shuffle, compress_seq=True, preload=True, num_workers=8, collate_fn=NERDataset.collate_fn, **kwargs):
    dataset = NERDataset(path=path, tag2id=tag2id, tokenizer=tokenizer, preload=preload, **kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, compress_seq))
    return data_loader