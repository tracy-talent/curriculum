"""
 Author: liujian
 Date: 2021-04-22 11:41:38
 Last Modified by: liujian
 Last Modified time: 2021-04-22 11:41:38
"""

from metrics import Mean, micro_p_r_f1_score
from utils import extract_kvpairs_in_bio, extract_kvpairs_in_bmoe
from data_loader import NERDataLoader

import os
from collections import defaultdict

import torch
from torch import nn, optim
from transformers import get_cosine_schedule_with_warmup


class NER_Trainer(nn.Module):
    """model(adaptive) + crf decoder"""
    
    def __init__(self, 
                model, 
                train_path, 
                val_path, 
                test_path, 
                ckpt, 
                logger,
                compress_seq=True,
                tagscheme='bmoes', 
                warmup_epoches=0,
                batch_size=32, 
                max_epoch=100, 
                lr=1e-3,
                metric='micro_f1',
                opt='adam'):

        super().__init__()

        # Load Data
        if train_path != None:
            self.train_loader = NERDataLoader(
                path=train_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=True,
                compress_seq=compress_seq
            )

        if val_path != None:
            self.val_loader = NERDataLoader(
                path=val_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )
        
        if test_path != None:
            self.test_loader = NERDataLoader(
                path=test_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )

        self.model = model
        self.parallel_model = nn.DataParallel(model)
        self.max_epoch = max_epoch
        self.metric = metric
        self.tagscheme = tagscheme
        self.lr = lr

        # scheduler
        self.warmup_epoches = warmup_epoches
        if warmup_epoches > 0:
            training_steps = len(self.train_loader) // batch_size * self.max_epoch
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=len(self.train_loader) // batch_size * warmup_epoches,
                                                             num_training_steps=training_steps)
        # loss
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # optimizer
        if opt == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.05)
        elif opt == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        elif opt == 'adamw': # Optimizer for BERT
            self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"{opt} optimizer is not supported!")
        # cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt
        # logger
        self.logger = logger


    def train_model(self):
        global_step = 0
        val_best_metric = 0
        negid = -1
        if 'O' in self.model.tag2id:
            negid = self.model.tag2id['O']
        if negid == -1:
            raise Exception("negative tag not is 'O'")

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            preds_kvpairs = []
            golds_kvpairs = []
            avg_loss = Mean()
            avg_acc = Mean()
            prec = Mean()
            rec = Mean()
            for ith, data in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        data[i] = data[i].cuda()
                args = data[1:]
                logits = self.parallel_model(*args)
                outputs_seq = data[0]
                inputs_seq, inputs_mask = data[1], data[-1]
                inputs_seq_len = inputs_mask.sum(dim=-1)
                bs = outputs_seq.size(0)

                # Optimize
                if self.model.crf is None:
                    loss = self.criterion(logits.permute(0, 2, 1), outputs_seq) # B * S
                    loss = torch.sum(loss * inputs_mask, dim=-1) / inputs_seq_len # B
                else:
                    log_likelihood = self.model.crf(logits, outputs_seq, mask=inputs_mask, reduction='none')
                    loss = -log_likelihood / inputs_seq_len
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                if self.warmup_epoches > 0:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # prediction/decode
                if self.model.crf is None:
                    preds_seq = logits.argmax(dim=-1) # B * S
                else:
                    preds_seq = self.model.crf.decode(logits, mask=inputs_mask) # List[List[int]]
                    for pred_seq in preds_seq:
                        pred_seq.extend([negid] * (outputs_seq.size(1) - len(pred_seq)))
                    preds_seq = torch.tensor(preds_seq).to(outputs_seq.device) # B * S
                
                # get token sequence
                preds_seq = preds_seq.detach().cpu().numpy()
                outputs_seq = outputs_seq.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_mask = inputs_mask.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    pred_seq_tag = [self.model.id2tag[tid] for tid in preds_seq[i][:seqlen]]
                    gold_seq_tag = [self.model.id2tag[tid] for tid in outputs_seq[i][:seqlen]]
                    char_seq = [self.model.sequence_encoder.id2token[int(tid)] for tid in inputs_seq[i][:seqlen]]

                    pred_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}')(pred_seq_tag, char_seq)
                    gold_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}')(gold_seq_tag, char_seq)

                    preds_kvpairs.append(pred_kvpairs)
                    golds_kvpairs.append(gold_kvpairs)

                # metrics update
                p_sum = 0
                r_sum = 0
                hits = 0
                for pred, gold in zip(preds_kvpairs[-bs:], golds_kvpairs[-bs:]):
                    p_sum += len(pred)
                    r_sum += len(gold)
                    for label in pred:
                        if label in gold:
                            hits += 1
                acc = ((outputs_seq == preds_seq) * (outputs_seq != negid) * inputs_mask).sum()

                # Log
                avg_loss.update(loss.item() * bs, bs) # must call item to split it from tensor graph, otherwise gpu memory will overflow
                avg_acc.update(acc, ((outputs_seq != negid) * inputs_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
                global_step += 1
                if global_step % 5 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.logger.info(f'Training...Epoches: {epoch}, steps: {global_step}, loss: {avg_loss.avg:.4f}, acc: {avg_acc.avg:.4f}, micro_p: {prec.avg:.4f}, micro_r: {rec.avg:.4f}, micro_f1: {micro_f1:.4f}')

            # Val 
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader) 
            self.logger.info(f'Evaluation result: {result}.')
            self.logger.info('Metric {} current / best: {} / {}'.format(self.metric, result[self.metric], val_best_metric))
            if result[self.metric] > val_best_metric:
                self.save_model(self.ckpt)
                val_best_metric = result[self.metric]
                self.logger.info("Best ckpt and saved.")
            
        self.logger.info("Best %s on val set: %f" % (self.metric, val_best_metric))


    def eval_model(self, eval_loader):
        self.eval()
        preds_kvpairs = []
        golds_kvpairs = []
        category_result = defaultdict(lambda: [0, 0, 0]) # gold, pred, correct
        avg_loss = Mean()
        avg_acc = Mean()
        prec = Mean()
        rec = Mean()
        if 'O' in self.model.tag2id:
            negid = self.model.tag2id['O']
        if negid == -1:
            raise Exception("negative tag not in 'O'")
        with torch.no_grad():
            for ith, data in enumerate(eval_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        data[i] = data[i].cuda()
                args = data[1:]
                logits = self.parallel_model(*args)
                outputs_seq = data[0]
                inputs_seq, inputs_mask = data[1], data[-1]
                inputs_seq_len = inputs_mask.sum(dim=-1)
                bs = outputs_seq.size(0)

                # loss
                if self.model.crf is None:
                    loss = self.criterion(logits.permute(0, 2, 1), outputs_seq) # B * S
                    loss = torch.sum(loss * inputs_mask, dim=-1) / inputs_seq_len # B
                else:
                    log_likelihood = self.model.crf(logits, outputs_seq, mask=inputs_mask, reduction='none')
                    loss = -log_likelihood / inputs_seq_len
                loss = loss.sum().item()

                # prediction/decode
                if self.model.crf is None:
                    preds_seq = logits.argmax(-1) # B * S
                else:
                    preds_seq = self.model.crf.decode(logits, mask=inputs_mask) # List[List[int]]
                    for pred_seq in preds_seq:
                        pred_seq.extend([negid] * (outputs_seq.size(1) - len(pred_seq)))
                    preds_seq = torch.tensor(preds_seq).to(outputs_seq.device) # B * S

                # get token sequence
                preds_seq = preds_seq.detach().cpu().numpy()
                outputs_seq = outputs_seq.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_mask = inputs_mask.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    pred_seq_tag = [self.model.id2tag[tid] for tid in preds_seq[i][:seqlen]]
                    gold_seq_tag = [self.model.id2tag[tid] for tid in outputs_seq[i][:seqlen]]
                    char_seq = [self.model.sequence_encoder.id2token[int(tid)] for tid in inputs_seq[i][:seqlen]]
                    
                    pred_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}')(pred_seq_tag, char_seq)
                    gold_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}')(gold_seq_tag, char_seq)

                    preds_kvpairs.append(pred_kvpairs)
                    golds_kvpairs.append(gold_kvpairs)

                # metrics update
                p_sum = 0
                r_sum = 0
                hits = 0
                for pred, gold in zip(preds_kvpairs[-bs:], golds_kvpairs[-bs:]):
                    for triple in gold:
                        category_result[triple[1]][0] += 1
                    for triple in pred:
                        category_result[triple[1]][1] += 1
                    p_sum += len(pred)
                    r_sum += len(gold)
                    for triple in pred:
                        if triple in gold:
                            hits += 1
                            category_result[triple[1]][2] += 1
                acc = ((outputs_seq == preds_seq) * (outputs_seq != negid) * inputs_mask).sum()
                avg_acc.update(acc, ((outputs_seq != negid) * inputs_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
                avg_loss.update(loss, bs)

                # Log
                if (ith + 1) % 20 == 0:
                    self.logger.info(f'Evaluation...Batches: {ith + 1} finished')

        for k, v in category_result.items():
            v_golden, v_pred, v_correct = v
            cate_precision = 0 if v_pred == 0 else round(v_correct / v_pred, 4)
            cate_recall = 0 if v_golden == 0 else round(v_correct / v_golden, 4)
            if cate_precision + cate_recall == 0:
                cate_f1 = 0
            else:
                cate_f1 = round(2 * cate_precision * cate_recall / (cate_precision + cate_recall), 4)
            category_result[k] = (cate_precision, cate_recall, cate_f1)
        category_result = {k: v for k, v in sorted(category_result.items(), key=lambda x: x[1][2])}
        p, r, f1 = micro_p_r_f1_score(preds_kvpairs, golds_kvpairs)
        result = {'loss': avg_loss.avg, 'acc': avg_acc.avg, 'micro_p': p, 'micro_r': r, 'micro_f1': f1, 'category-p/r/f1':category_result}
        return result

    def load_model(self, ckpt):
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict['model'])
    
    def save_model(self, ckpt):
        state_dict = {'model': self.model.state_dict()}
        torch.save(state_dict, ckpt)
