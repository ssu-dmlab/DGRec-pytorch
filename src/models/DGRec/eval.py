#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from models.DGRec.model import DGRec
from models.DGRec.batch.minibatch import MinibatchIterator
from tqdm import tqdm
from torch.nn import functional as F

class MyEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, minibatch, hyper_param, mode='test'):
        with torch.no_grad():

            epochs = hyper_param['epochs']
            act = hyper_param['act']
            batch_size = hyper_param['batch_size']
            max_degree = hyper_param['max_degree']
            num_users = hyper_param['num_users']
            num_items = hyper_param['num_items']
            learning_rate = hyper_param['learning_rate']
            hidden_size = hyper_param['hidden_size']
            embedding_size = hyper_param['embedding_size']
            emb_user = hyper_param['emb_user']
            max_length = hyper_param['max_length']
            samples_1 = hyper_param['samples_1']
            samples_2 = hyper_param['samples_2']
            dim1 = hyper_param['dim1']
            dim2 = hyper_param['dim2']
            dropout = hyper_param['dropout']
            weight_decay = hyper_param['weight_decay']
            print_every = hyper_param['print_every']
            val_every = hyper_param['val_every']

            model.eval()

            minibatch.shuffle()
            feed_dict = minibatch.next_val_minibatch_feed_dict(mode)

            # evaluation
            labels = torch.tensor(np.array(feed_dict['output_session']), dtype=torch.long) # labels : [batch, max_length]
            predictions = model.predict(feed_dict) # predictions : [batch, max_length, item_embedding]

            loss = self._loss(predictions, labels)
            recall_k = self._recall(predictions, labels, batch_size)
            ndcg = self._ndcg(predictions, labels, num_items, feed_dict['mask_y'])

        return loss.item(), recall_k.item(), ndcg.item()

    def _loss(self, predictions, labels):
        logits = torch.swapaxes(predictions, 1, 2)
        logits = logits.to(dtype=torch.float)
        #print(logits.shape)
        labels = torch.tensor(np.array(labels), dtype=torch.long)

        loss = F.cross_entropy(logits,
                               labels)  # logits : [batch, item_embedding, max_length] / labels : [batch, max_length]

        return loss

    def _recall(self, predictions, labels, batch_size):
        _, top_k_index = torch.topk(predictions, k=20, dim=2)  # top_k_index : [batch, max_length, k]

        labels = torch.unsqueeze(labels, dim=2)  # labels : [batch, max_length, 1]
        corrects = (top_k_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]
        recall_corrects = torch.sum(corrects, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]

        mask_sum = (labels != 0).sum(dim=1)  # mask_sum : [batch, 1]
        mask_sum = torch.squeeze(mask_sum, dim=1)  # mask_sum : [batch]

        recall_k = (recall_corrects.sum(dim=1) / mask_sum).sum() / batch_size

        return recall_k

    def _ndcg(self, logits, labels, num_items, mask):
        batch_size = logits.shape[0]
        logits = torch.reshape(logits, (logits.shape[0] * logits.shape[1], logits.shape[2]))
        predictions = torch.transpose(logits, 0, 1)
        targets = torch.reshape(labels, [-1])
        pred_values = torch.unsqueeze(torch.diagonal(predictions[targets]), -1)
        #print(pred_values)
        #print(targets)
        tile_pred_values = torch.tile(pred_values, [1, num_items])
        #print(tile_pred_values.shape)
        #print(tile_pred_values)
        #print(logits)
        ranks = torch.sum((logits > tile_pred_values).type(torch.float), -1) + 1
        #print(ranks)
        ndcg = 1. / (torch.log2(1.0 + ranks))

        mask = torch.Tensor(mask)
        mask = torch.reshape(mask, [-1])
        ndcg *= mask

        return torch.sum(ndcg) / batch_size
