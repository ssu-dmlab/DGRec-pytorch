#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from loguru import logger


class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., bias=False, act=nn.ReLU(), **kwargs):
        super().__init__()

        self.act = act

        self.feat_drop = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.fc.weight = torch.nn.init.xavier_uniform_(self.fc.weight)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        self_vecs, neigh_vecs = inputs

        if self.feat_drop is not None:
            self_vecs = self.feat_drop(self_vecs)
            neigh_vecs = self.feat_drop(neigh_vecs)

        # Reshape from [batch_size, depth] to [batch_size, 1, depth] for matmul.
        self_vecs = torch.unsqueeze(self_vecs, 1)  # [batch, 1, embedding_size]
        neigh_self_vecs = torch.cat((neigh_vecs, self_vecs), dim=1)  # [batch, sample, embedding]

        score = self.softmax(torch.matmul(self_vecs, torch.transpose(neigh_self_vecs, 1, 2)))

        # alignment(score) shape is [batch_size, 1, depth]
        context = torch.squeeze(torch.matmul(score, neigh_self_vecs), dim=1)

        # [nodes] x [out_dim]
        output = self.act(self.fc(context))

        return output


class DGRec(torch.nn.Module):
    def __init__(
            self,
            hyper_param,
            num_layers,
    ):
        super(DGRec, self).__init__()
        self.act = hyper_param['act']
        self.batch_size = hyper_param['batch_size']
        self.num_users = hyper_param['num_users']
        self.num_items = hyper_param['num_items']
        self.embedding_size = hyper_param['embedding_size']
        self.max_length = hyper_param['max_length']
        self.samples_1 = hyper_param['samples_1']
        self.samples_2 = hyper_param['samples_2']
        self.dropout = hyper_param['dropout']
        self.num_layers = num_layers

        if self.act == 'relu':
            self.act = nn.ReLU()
        elif self.act == 'elu':
            self.act = nn.ELU()

        self.user_embedding = nn.Embedding(self.num_users,
                                           self.embedding_size)  # (num_users=26511, embedding_dim=100)
        self.item_embedding = nn.Embedding(self.num_items,
                                           self.embedding_size,
                                           padding_idx=0)  # (num_items=12592, embedding_dim=100)
        self.item_indices = nn.Parameter(torch.arange(0, self.num_items, dtype=torch.long),
                                         requires_grad=False)

        self.feat_drop = nn.Dropout(self.dropout) if self.dropout > 0 else None
        input_dim = self.embedding_size

        # making user embedding
        self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)

        # combine friend's long and short-term interest
        self.W1 = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=False)
        self.W1.weight = torch.nn.init.xavier_uniform_(self.W1.weight)

        # combine user interest and social influence
        self.W2 = nn.Linear(input_dim + self.embedding_size, self.embedding_size, bias=False)
        self.W2.weight = torch.nn.init.xavier_uniform_(self.W2.weight)

        # making GAT layers
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            aggregator = GAT(input_dim, input_dim, act=self.act, dropout=self.dropout)
            self.layers.append(aggregator)

    # get target user's interest
    def individual_interest(self, input_session):
        input = input_session[0].long()  # input.shape : [max_length]
        emb_seqs = self.item_embedding(input)  # emb_seqs.shape : [max_length, embedding_dim]
        emb_seqs = torch.unsqueeze(emb_seqs, 0)

        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)

        for batch in range(self.batch_size - 1):
            input = input_session[batch + 1].long()
            emb_seq = self.item_embedding(input)
            emb_seq = torch.unsqueeze(emb_seq, 0)
            emb_seqs = torch.cat((emb_seqs, emb_seq), 0)

        hu, (_, _) = self.lstm(emb_seqs)  # output.shape : [batch_size, max_length, embedding_dim]

        return hu

    # get friends' interest
    def friends_interest(self, support_nodes_layer1, support_nodes_layer2, support_sessions_layer1,
                         support_sessions_layer2):
        ''' long term '''
        long_term = []
        support_nodes_layer = [support_nodes_layer1, support_nodes_layer2]

        for layer in support_nodes_layer:
            long_input = layer.long()
            long_term1_2 = self.user_embedding(long_input)
            long_term.append(long_term1_2)
            # long_term[0].shape : [sample1 * sample2, embedding_dim]
            # long_term[1].shape : [sample2, embedding_dim]

        ''' short term '''
        short_term = []
        support_sessions_layer = [support_sessions_layer1, support_sessions_layer2]
        sample1_2 = [self.samples_1 * self.samples_2, self.samples_2]

        for layer, sample in zip(support_sessions_layer, sample1_2):
            short_arange = torch.arange(self.batch_size * sample, dtype=torch.long)
            short_input = layer[short_arange].long()
            friend_emb_seqs = self.item_embedding(short_input)

            if self.feat_drop is not None:
                friend_emb_seqs = self.feat_drop(friend_emb_seqs)

            _, (_, short_term1_2) = self.lstm(friend_emb_seqs)
            short_term1_2 = torch.squeeze(short_term1_2)

            short_term.append(short_term1_2)
            # short_term[0].shape : [batch_size * sample1 * sample2, embedding_dim]
            # short_term[1].shape : [batch_size * sample2, embedding_dim]

        ''' long-short term'''
        long_short_term1 = torch.cat((long_term[0], short_term[0]), dim=1)
        long_short_term2 = torch.cat((long_term[1], short_term[1]), dim=1)
        # long_short_term1.shape : [batch_size * sample1 * sample2, embedding_dim + embedding_dim]
        # long_short_term2.shape : [batch_size * sample2, embedding_dim + embedding_dim]

        if self.feat_drop is not None:
            long_short_term1 = self.feat_drop(long_short_term1)
            long_short_term2 = self.feat_drop(long_short_term2)

        long_short_term1 = torch.relu(self.W1(long_short_term1))
        long_short_term2 = torch.relu(self.W1(long_short_term2))
        # long_short_term1.shape : [batch_size * sample1 * sample2, embedding_dim]
        # long_short_term2.shape : [batch_size * sample2, embedding_dim]

        long_short_term = [long_short_term2, long_short_term1]

        return long_short_term

    # get user's interest influenced by friends
    def social_influence(self, hu, long_short_term):
        hu = torch.transpose(hu, 0, 1)
        outputs = []
        support_sizes = [1, self.samples_2, self.samples_1 * self.samples_2]
        num_samples = [self.samples_1, self.samples_2]
        for i in range(self.max_length):
            count = 0
            hu_ = hu[i]  # implement 1 of 20
            hidden = [hu_, long_short_term[0], long_short_term[1]]
            for layer in self.layers:
                next_hidden = []
                for hop in range(self.num_layers - count):
                    neigh_dims = [self.batch_size * support_sizes[hop],
                                  num_samples[self.num_layers - hop - 1],
                                  self.embedding_size]
                    h = layer([hidden[hop],
                               torch.reshape(hidden[hop + 1], neigh_dims)])
                    next_hidden.append(h)
                hidden = next_hidden
                count += 1
            outputs.append(hidden[0])
        feat = torch.stack(outputs, axis=0)
        # hu.shape, feat.shape : [max_length, batch, embedding_size]

        sr = self.W2(torch.cat((hu, feat), dim=2))  # final representation

        # return : [batch, max_length, item_embedding]
        return sr

    # get item score
    def score(self, sr, mask_y):
        logits = sr @ self.item_embedding(self.item_indices).t()  # similarity
        # logit shape : [max_length, batch, item_embedding]

        mask = mask_y.long()
        logits = torch.transpose(logits, 0, 1)
        logits *= torch.unsqueeze(mask, 2)

        return logits

    def forward(self, feed_dict):
        '''
        * Individual interest
            - Input_x: Itemid that user consumed in Timeid(session) - input data
                [batch_size, max_length]
            - Input_y: Itemid that user consumed in Timeid(session) - label
                [batch_size, max_length]
            - mask_y: mask of input_y
                [batch_size, max_length]
        * Friends' interest (long-term)
            - support_nodes_layer1: Userid of friends' friends
                [batch_size * samples_1 * samples_2]
            - support_nodes_layer2: Userid of friends
                [batch_size * samples_2]
        * Friends' interest (short-term)
            - support_sessions_layer1: Itemid that friends' friends spent most recently on Timeid.
                [batch_size * samples_1 * samples_2]
            - support_sessions_layer2: Itemid that friends spent most recently on Timeid.
                [batch_size * samples_2]
            - support_lengths_layer1: Number of items consumed by support_sessions_layer1
                [batch_size * samples_1 * samples_2]
            - support_lengths_layer2: Number of items consumed by support_sessions_layer2
                [batch_size * samples_2]
        '''
        labels = feed_dict['output_session']

        # interest
        hu = self.individual_interest(feed_dict['input_session'])

        long_short_term = self.friends_interest(feed_dict['support_nodes_layer1'],
                                                feed_dict['support_nodes_layer2'],
                                                feed_dict['support_sessions_layer1'],
                                                feed_dict['support_sessions_layer2'])

        # social influence
        sr = self.social_influence(hu, long_short_term)

        # score
        logits = self.score(sr, feed_dict['mask_y'])

        # metric
        recall = self._recall(logits, labels)
        ndcg = self._ndcg(logits, labels, feed_dict['mask_y'])

        # loss
        logits = (torch.transpose(logits, 1, 2)).to(dtype=torch.float)  # logits : [batch, item_embedding, max_length]
        labels = labels.long()  # labels : [batch, max_length]

        loss = F.cross_entropy(logits, labels)

        return loss, recall, ndcg  # loss, recall_k, ndcg

    def predict(self, feed_dict):
        labels = feed_dict['output_session']

        hu = self.individual_interest(feed_dict['input_session'])

        long_short_term = self.friends_interest(feed_dict['support_nodes_layer1'],
                                                feed_dict['support_nodes_layer2'],
                                                feed_dict['support_sessions_layer1'],
                                                feed_dict['support_sessions_layer2'])

        sr = self.social_influence(hu, long_short_term)

        logits = self.score(sr, feed_dict['mask_y'])

        # metric
        recall = self._recall(logits, labels)
        ndcg = self._ndcg(logits, labels, feed_dict['mask_y'])

        # loss
        logits = (torch.transpose(logits, 1, 2)).to(dtype=torch.float)  # logits : [batch, item_embedding, max_length]
        labels = labels.long()  # labels : [batch, max_length]

        loss = F.cross_entropy(logits, labels)

        return loss, recall, ndcg

    def _recall(self, predictions, labels):
        batch_size = predictions.shape[0]
        _, top_k_index = torch.topk(predictions, k=20, dim=2)  # top_k_index : [batch, max_length, k]

        labels = labels.long()
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

        labels = labels.long()
        targets = torch.reshape(labels, [-1])
        pred_values = torch.unsqueeze(torch.diagonal(predictions[targets]), -1)
        # tile_pred_values = torch.tile(pred_values, [1, num_items])
        tile_pred_values = pred_values.repeat(1, num_items)
        ranks = torch.sum((logits > tile_pred_values).type(torch.float), -1) + 1
        ndcg = 1. / (torch.log2(1.0 + ranks))

        mask_sum = torch.sum(mask)
        mask = torch.reshape(mask, [-1])
        ndcg *= mask

        return torch.sum(ndcg) / mask_sum