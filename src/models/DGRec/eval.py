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

            minibatch.val_shuffle()
            feed_dict = minibatch.next_val_minibatch_feed_dict(mode)

            # evaluation
            loss, recall_k, ndcg = model.predict(feed_dict)

        return loss.item(), recall_k.item(), ndcg.item()


