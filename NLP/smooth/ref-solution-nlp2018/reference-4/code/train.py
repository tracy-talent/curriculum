import collections
import logging
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import time

from metrics import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from pytorch_pretrained_bert.optimization import BertAdam

class Trainer(object):

    def __init__(self, args, model, train_examples, use_gpu):
        self.use_gpu = use_gpu
        self.model = model

        self.epochs = args.epochs
        self.best_f1 = -1
        self.min_loss = 100
        self.save_dir = args.save_dir

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.lr = args.lr
        self.warmup_proportion = args.warmup_proportion
        self.t_total = int(train_examples / args.batch_size / 1 * args.epochs)

        self.optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=args.lr,
                                  warmup=args.warmup_proportion,
                                  t_total=self.t_total)

        if self.use_gpu:
            self.loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, args.weight]).cuda())
        else:
            self.loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, args.weight]))


    def train(self, train_dataloader, dev_dataloader):

        global_step = 0

        for epoch in range(self.epochs):
            self.model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            start = time.time()

            for index, (ids, type, mask, label) in enumerate(train_dataloader):

                if self.use_gpu:
                    ids, type, mask, label = ids.cuda(), type.cuda(), mask.cuda(), label.cuda()

                output = self.model(ids, type, mask)
                loss = self.loss_func(output, label)
                # loss = self.model(ids, type, mask, label)

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += ids.size(0)
                nb_tr_steps += 1

                if (index + 1) % 1 == 0:
                    lr_this_step = self.lr * self.warmup_linear(global_step / self.t_total, self.warmup_proportion)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

            self.val(dev_dataloader, epoch + 1, tr_loss / nb_tr_examples, start)

    def val(self, val_dataloader, epoch, loss, start_time):
        ground_truth = []
        prediction = []
        self.model.eval()
        for index, (ids, type, mask, label) in enumerate(val_dataloader):
            if self.use_gpu:
                ids, type, mask = ids.cuda(), type.cuda(), mask.cuda()

            with torch.no_grad():
                logits = self.model(ids, type, mask)

            pred = torch.argmax(logits, 1).detach().cpu().numpy().astype(np.int32)
            label = label.detach().numpy().astype(np.int32)

            ground_truth.extend(label)
            prediction.extend(pred)

        precision = precision_score(ground_truth, prediction)
        recall = recall_score(ground_truth, prediction)
        f1 = f1_score(ground_truth, prediction)

        logging.info("Epoch: {} | Loss: {:.5f} | Precision: {:.5f} | Recall: {:.5f} | F1: {:.5f} | Time: {:.3f}"
                     .format(epoch, loss, precision, recall, f1, (time.time() - start_time) / 60))
        # logging.info("Epoch: {} | Loss: {:.5f} | Time: {:.3f}"
        #              .format(epoch, loss, (time.time() - start_time) / 60))

        # if f1 > self.best_f1:
        #     self.best_f1 = f1
        # if loss < self.min_loss:
        #     self.min_loss = loss
        logging.info("Saving checkpoint...")
        state_dict = self.model.state_dict()
        save_path = os.path.join(self.save_dir, "ckpt-epoch-{}".format(epoch))
        torch.save(state_dict, save_path)
        logging.info("Checkpoint saved to {}.".format(save_path))

        # return precision, recall, f1

    def predict(self, test_dataloader):
        prediction = []
        id = []
        self.model.eval()
        for index, (ids, type, mask, sentence_id) in enumerate(test_dataloader):
            if self.use_gpu:
                ids, type, mask = ids.cuda(), type.cuda(), mask.cuda()

            with torch.no_grad():
                output = self.model(ids, type, mask)

            pred = torch.argmax(output, 1).detach().cpu().numpy().astype(np.int32)
            prediction.extend(pred)
            id.extend(sentence_id)

        return prediction, id

    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x
