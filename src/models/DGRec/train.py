#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from models.DGRec.model import DGRec
from models.DGRec.eval import MyEvaluator
from models.DGRec.batch.minibatch import MinibatchIterator
from tqdm import tqdm
from loguru import logger

class MyTrainer:
    def __init__(self, device):
        self.device = device
        self.train_losses = []
        self.train_recall = []
        self.train_ndcg = []
        self.val_losses = []
        self.val_recall = []
        self.val_ndcg = []

    def train_with_hyper_param(self, minibatch, hyper_param):
        seed = hyper_param['seed']
        epochs = hyper_param['epochs']
        learning_rate = hyper_param['learning_rate']
        data_name = hyper_param['data_name']
        embedding_size = hyper_param['embedding_size']
        decay_rate = hyper_param['decay_rate']

        model = DGRec(hyper_param, num_layers=2).to(self.device)
        evaluator = MyEvaluator(device=self.device)

        patience = 20
        inc = 0
        early_stopping = False
        highest_val_ndcg = 0

        batch_len = minibatch.train_batch_len()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=batch_len // 10, gamma=decay_rate)

        pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')

        for epoch in pbar:
            total_loss = 0
            total_recall = 0
            total_ndcg = 0

            minibatch.shuffle()

            for batch in tqdm(range(batch_len), position=1, leave=False, desc='batch'):
                model.train()
                optimizer.zero_grad()

                feed_dict = minibatch.next_train_minibatch_feed_dict()

                # train
                loss, recall_k, ndcg = model(feed_dict)

                loss.backward()
                optimizer.step()
                scheduler.step()

                # log
                total_loss += loss.item()
                total_recall += recall_k
                total_ndcg += ndcg
                self.train_recall.append(recall_k)
                self.train_ndcg.append(ndcg)

                # validation
                if (batch % int(batch_len / 10)) == 0:
                    val_loss, val_recall_k, val_ndcg = evaluator.evaluate(model, minibatch, mode='val')

                    self.val_recall.append(val_recall_k)
                    self.val_ndcg.append(val_ndcg)

                    if val_ndcg >= highest_val_ndcg:
                        highest_val_ndcg = val_ndcg
                        inc = 0
                    else:
                        inc += 1

                # early stopping
                if inc >= patience:
                    early_stopping = True
                    break

            if early_stopping:
                pbar.write('Early stop at epoch: {}, batch steps: {}'.format(epoch+1, batch))
                pbar.update(pbar.total)
                break

            pbar.write(
                'Epoch {:02}: train loss: {:.4}\t  train recall@20: {:.4}\t  train NDCG: {:.4}'
                .format(epoch+1, total_loss/batch_len, total_recall/batch_len, total_ndcg/batch_len))
            pbar.write(
                'Epoch {:02}: valid loss: {:.4}\t  valid recall@20: {:.4}\t  valid NDCG: {:.4}\n'
                .format(epoch+1, val_loss, val_recall_k, val_ndcg))
            pbar.update()

        pbar.close()

        # plot training metric graph
        plt.figure(2, figsize=(10, 5))
        plt.title(" Training metric")
        plt.plot(self.train_recall, label="recall")
        plt.plot(self.train_ndcg, label="ndcg")
        plt.xlabel("time step (=iterations)")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(' training metric --data_name ' + str(data_name) + ' --seed ' + str(seed) + ' --emb ' + str(embedding_size) + '.png')
        plt.clf()

        # plot validation metric graph
        plt.figure(3, figsize=(10, 5))
        plt.title(" Validation metric")
        plt.plot(self.val_recall, label="recall")ß
        plt.plot(self.val_ndcg, label="ndcg")
        plt.xlabel("time step (=batch_size/10)")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(' validation metric --data_name ' + str(data_name) + ' --seed ' + str(seed) + ' --emb ' + str(embedding_size) + '.png')
        plt.clf()

        return model
