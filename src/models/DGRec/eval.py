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
        val_loss = []
        val_recall = []
        val_ndcg = []
        with torch.no_grad():
            model.eval()

            while not minibatch.end_val(mode):
                feed_dict = minibatch.next_val_minibatch_feed_dict(mode)
                loss, recall_k, ndcg = model.predict(feed_dict)

                val_loss.append(loss.item())
                val_recall.append(recall_k)
                val_ndcg.append(ndcg)

        return np.mean(val_loss), np.mean(val_recall), np.mean(val_ndcg)

