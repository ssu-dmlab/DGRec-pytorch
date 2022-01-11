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

    #def train_with_hyper_param(self, minibatch, hyper_param, val_minibatch=None):
    def train_with_hyper_param(self, minibatch, hyper_param):
        seed = hyper_param['seed']
        epochs = hyper_param['epochs']
        learning_rate = hyper_param['learning_rate']

        model = DGRec(hyper_param, num_layers=2).to(self.device)
        evaluator = MyEvaluator(device=self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.98)

        patience = 10
        inc = 0
        early_stopping = False
        lowest_val_loss = float('inf')

        batch_len = minibatch.train_batch_len()

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

                total_loss += loss.item()
                total_recall += recall_k.item()
                total_ndcg += ndcg.item()

                self.train_losses.append(loss.item())
                self.train_recall.append(recall_k.item())
                self.train_ndcg.append(ndcg.item())

                # validation
                val_loss, val_recall_k, val_ndcg = evaluator.evaluate(model, minibatch, mode='val')

                self.val_losses.append(val_loss)
                self.val_recall.append(val_recall_k)
                self.val_ndcg.append(val_ndcg)

                if (batch % 100) == 0:
                    if(val_loss <= lowest_val_loss):
                        lowest_val_loss = val_loss
                        inc = 0
                    else:
                        inc += 1

                if inc >= patience:
                    early_stopping = True
                    break

            if early_stopping:
                print('Early stop at epoch: {}, batch steps: {}'.format(epoch+1, batch))
                break

            pbar.write(
                'Epoch {:02}: training loss: {:.4},  training recall@20: {:.4},  training NDCG: {:.4}'
                .format(epoch+1, total_loss/batch_len, total_recall/batch_len, total_ndcg/batch_len))
            pbar.write(
                'Epoch {:02}: valid loss: {:.4},  valid recall@20: {:.4},  valid NDCG: {:.4}'
                .format(epoch+1, val_loss, val_recall_k, val_ndcg))
            pbar.update()

        pbar.close()

        # plot loss graph
        plt.figure(1, figsize=(10, 5))
        plt.title(" Training and Validation Loss")
        plt.plot(self.train_losses, label="train")
        plt.plot(self.val_losses, label="val")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(' loss - seed:' + str(seed) + '.png')

        # plot metric graph
        plt.figure(2, figsize=(10, 5))
        plt.title(" Training and Validation recall@20")
        plt.plot(self.train_recall, label="train")
        plt.plot(self.val_recall, label="val")
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(' metric(recall@20) - seed:' + str(seed) + '.png')

        plt.figure(3, figsize=(10, 5))
        plt.title(" Training and Validation NDCG")
        plt.plot(self.train_ndcg, label="train")
        plt.plot(self.val_ndcg, label="val")
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(' metric(ndcg) - seed:' + str(seed) + '.png')

        return model
