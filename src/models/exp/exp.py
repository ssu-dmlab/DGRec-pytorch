#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from loguru import logger


class ItemModel(torch.nn.Module):
    def __init__(
            self,
    ):
        super(ItemModel, self).__init__()
        self.user_embedding = nn.Embedding(10558,
                                           100)  # (num_users=26511, embedding_dim=100)
        self.item_embedding = nn.Embedding(2968,
                                           100,
                                           padding_idx=0)  # (num_items=12592, embedding_dim=100)

        self.W1 = nn.Linear(100, 100, bias=False)
        self.W1.weight = torch.nn.init.xavier_uniform_(self.W1.weight)
        self.W2 = torch.nn.Softmax(dim=1)
        self.data = pd.read_csv('/Users/kimtaesu/PycharmProjects/DGRec-pytorch/datasets/bookdata/train.tsv', sep='\t',
                            dtype={0: np.int32, 1: np.int32, 2: np.int32})
        #print(self.data.iloc[[-1]])
        self.UserId = self.data['UserId']
        self.ItemId = self.data['ItemId']
        self.Rating = self.data['Rating']
        print(len(self.Rating)) #195665

    def forward(self):
        self.new_item_embedding = self.W2(self.W1(self.item_embedding.weight))
        batch_size = 500
        ItemId = self.ItemId[: batch_size]
        UserId = self.UserId[: batch_size]
        Rating = self.Rating[: batch_size]

        # consider negative value of inner product
        similarity = self.new_item_embedding[ItemId] @ self.user_embedding.weight[UserId].T
        loss = sum(-Rating[i] * similarity[i][i] for i in range(batch_size))

        return loss

def main():
    epochs = 20
    model = ItemModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')

    for epoch in pbar:
        for batch in tqdm(range(100), position=1, leave=False, desc='batch'):
            model.train()
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()

        pbar.write('Epoch {:02}: {:.4} training loss'.format(epoch, loss.item()))
        pbar.update()
    pbar.close()


if __name__ == "__main__":
    main()
