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
        self.val_losses = []
        self.val_recall = []
        self.val_ndcg = []

    def train_with_hyper_param(self, minibatch, hyper_param, val_minibatch=None):
        seed = hyper_param['seed']
        device = hyper_param['device']
        epochs = hyper_param['epochs']
        learning_rate = hyper_param['learning_rate']

        model = DGRec(hyper_param, num_layers=2).to(self.device)
        evaluator = MyEvaluator(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.98)

        patience = 20
        inc = 0
        early_stopping = False
        highest_val_recall = -1.0

        batch_len = minibatch.train_batch_len()
        batch_len = int(batch_len)

        pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')

        for epoch in pbar:
            minibatch.shuffle()
            for batch in tqdm(range(batch_len), position=1, leave=False, desc='batch'):

                model.train()
                feed_dict = minibatch.next_train_minibatch_feed_dict()
                optimizer.zero_grad()

                loss = model(feed_dict, feed_dict['output_session'])
                self.train_losses.append(loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()

                val_loss, val_recall_k, val_ndcg = evaluator.evaluate(model, val_minibatch, 'val')
                self.val_losses.append(val_loss)
                self.val_recall.append(val_recall_k)
                self.val_ndcg.append(val_ndcg)

                if (batch % 100) == 0:
                    print('Batch {:03}: train loss: {:.4} '.format(batch, loss.item()))

                    if (val_recall_k >= highest_val_recall):
                        pbar.write('Batch {:03}: valid loss: {:.4},  valid recall@20: {:.4},  valid ndcg: {:.4}'
                                   .format(batch, val_loss, val_recall_k, val_ndcg))
                        highest_val_recall = val_recall_k
                        inc = 0
                    else:
                        inc += 1

                if inc >= patience:
                    early_stopping = True
                    break

            if early_stopping:
                print('Early stop at epoch: {}, batch steps: {}'.format(epoch, batch))
                break

            pbar.write('Epoch {:02}: {:.4} training loss'.format(epoch, loss.item()))
            pbar.update()

        pbar.close()

        # plot graph
        plt.figure(1, figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(self.train_losses, label="train")
        plt.plot(self.val_losses, label="val")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('loss-' + str(seed) + '.png')

        plt.figure(2, figsize=(10, 5))
        plt.title("recall@20 and ndcg")
        plt.plot(self.val_recall, label="recall@20")
        plt.plot(self.val_ndcg, label="ndcg")
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.legend()
        #plt.show()
        plt.savefig('metric-' + str(seed) + '.png')

        return model
