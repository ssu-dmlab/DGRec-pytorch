#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from models.DGRec.model import DGRec
from models.DGRec.batch.minibatch import MinibatchIterator
from tqdm import tqdm
import torch.nn.functional as F

class MyEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, minibatch, mode='test'):
        with torch.no_grad():
            model.eval()

            minibatch.shuffle()
            feed_dict = minibatch.next_val_minibatch_feed_dict(mode)

            # evaluation
            labels = torch.tensor(np.array(feed_dict['output_session']), dtype=torch.long) # labels : [batch, max_length]
            predictions = model.predict(feed_dict) # predictions : [batch, max_length, item_embedding]

            loss = self._loss(predictions, labels)
            recall_k = self._recall(predictions, labels)
            ndcg = self._ndcg(predictions, labels, feed_dict['mask_y'])

        return loss.item(), recall_k.item(), ndcg.item()

    def _loss(self, predictions, labels):
        logits = torch.swapaxes(predictions, 1, 2)
        logits = logits.to(dtype=torch.float)
        labels = torch.tensor(np.array(labels), dtype=torch.long)

        loss = F.cross_entropy(logits,
                               labels)  # logits : [batch, item_embedding, max_length] / labels : [batch, max_length]

        return loss

    def _recall(self, predictions, labels):
        batch_size = predictions.shape[0]
        _, top_k_index = torch.topk(predictions, k=20, dim=2)  # top_k_index : [batch, max_length, k]

        labels = torch.unsqueeze(labels, dim=2)  # labels : [batch, max_length, 1]
        corrects = (top_k_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]
        recall_corrects = torch.sum(corrects, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]

        mask_sum = (labels != 0).sum(dim=1)  # mask_sum : [batch, 1]
        mask_sum = torch.squeeze(mask_sum, dim=1)  # mask_sum : [batch]

        recall_k = (recall_corrects.sum(dim=1) / mask_sum).sum()

        return recall_k / batch_size

    def _ndcg(self, logits, labels, mask):
        num_items = logits.shape[2]
        logits = torch.reshape(logits, (logits.shape[0] * logits.shape[1], logits.shape[2]))
        predictions = torch.transpose(logits, 0, 1)
        targets = torch.reshape(labels, [-1])
        pred_values = torch.unsqueeze(torch.diagonal(predictions[targets]), -1)
        tile_pred_values = torch.tile(pred_values, [1, num_items])
        ranks = torch.sum((logits > tile_pred_values).type(torch.float), -1) + 1
        ndcg = 1. / (torch.log2(1.0 + ranks))

        mask = torch.Tensor(mask)
        mask_sum = torch.sum(mask)
        mask = torch.reshape(mask, [-1])
        ndcg *= mask

        return torch.sum(ndcg) / mask_sum
