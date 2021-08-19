#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from models.DGRec.model import DGRec
from models.DGRec.eval import MyEvaluator
from models.DGRec.batch.minibatch import MinibatchIterator
from tqdm import tqdm
from loguru import logger

class MyTrainer:
    def __init__(self, device):
        self.device = device

    def train_with_hyper_param(self, minibatch, hyper_param):
        device = hyper_param['device']
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
        model_size = hyper_param['model_size']
        dropout = hyper_param['dropout']
        weight_decay = hyper_param['weight_decay']
        print_every = hyper_param['print_every']
        val_every = hyper_param['val_every']



        model = DGRec(hyper_param, num_layers=2).to(self.device)
        evaluator = MyEvaluator(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')

        batch_len = minibatch.train_batch_len()
        batch_len = int(batch_len)

        for epoch in pbar:
            minibatch.shuffle()
            for batch in tqdm(range(2), position=1, leave=False, desc='batch'):
                feed_dict = minibatch.next_train_minibatch_feed_dict()
                optimizer.zero_grad()

                loss = model(feed_dict, feed_dict['output_session'])

                loss.backward()

                optimizer.step()
                '''
                if (batch % 10) == 0:
                    accuracy, real_accuracy, recall_k = evaluator.evaluate(model, minibatch, hyper_param, 'val')

                    #pbar.write('Epoch {:02}: {:.4}  {:.4}\n'.format(epoch, accuracy, recall_k))
                    model.train()
                '''
            pbar.write('Epoch {:02}: {:.4} training loss'.format(epoch, loss.item()))
            pbar.update()

        pbar.close()

        return model
