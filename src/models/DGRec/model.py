#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from loguru import logger
from utils import glorot, zeros


class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, neigh_input_dim=None, dropout=0., bias=False, act=nn.ReLU(), concat=False, batch_norm=False, **kwargs):
        super().__init__()

        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        self.vars = {}
        self.vars['weights'] = glorot([neigh_input_dim, output_dim])

        if self.bias:
            self.vars['bias'] = zeros([self.output_dim])

        self.feat_drop = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(100, 100, bias=True)
        self.m = nn.Softmax(dim=-1)

        self.input_dim = input_dim
        self.output_dim = output_dim

        '''
        if batch_norm:
            self.batch_norm_q = nn.BatchNorm1d(self.input_dim)
            self.batch_norm_k = nn.BatchNorm1d(self.input_dim)
        else:
            self.batch_norm_q = None
            self.batch_norm_k = None
        '''
    def forward(self, inputs):
        self_vecs, neigh_vecs = inputs
        '''
        if self.batch_norm_q is not None:
            self_vecs = self.batch_norm_q(self_vecs)
            neigh_vecs = self.batch_norm_k(neigh_vecs)
        '''
        if self.feat_drop is not None:
            self_vecs = self.feat_drop(self_vecs)
            neigh_vecs = self.feat_drop(neigh_vecs)

        # Reshape from [batch_size, depth] to [batch_size, 1, depth] for matmul.
        query = torch.unsqueeze(self_vecs, 1) # [batch, 1, embedding_size]
        neigh_self_vecs = torch.cat((neigh_vecs, query), dim=1) # [batch, sample, embedding]
        score = torch.matmul(query, torch.transpose(neigh_self_vecs, 1, 2))
        score = self.m(score)

        # alignment(score) shape is [batch_size, 1, depth]
        context = torch.matmul(score, neigh_self_vecs)
        context = torch.squeeze(context, dim=1)

        # [nodes] x [out_dim]
        output = torch.matmul(context, self.vars['weights'])

        return self.act(output)


class DGRec(torch.nn.Module):
    def __init__(
            self,
            hyper_param,
            num_layers,
            feat_drop=0.,
    ):
        super(DGRec, self).__init__()
        self.epochs = hyper_param['epochs']
        self.act = hyper_param['act']
        self.batch_size = hyper_param['batch_size']
        self.max_degree = hyper_param['max_degree']
        self.num_users = hyper_param['num_users']
        self.num_items = hyper_param['num_items']
        self.learning_rate = hyper_param['learning_rate']
        self.hidden_size = hyper_param['hidden_size']
        self.embedding_size = hyper_param['embedding_size']
        self.emb_user = hyper_param['emb_user']
        self.max_length = hyper_param['max_length']
        self.samples_1 = hyper_param['samples_1']
        self.samples_2 = hyper_param['samples_2']
        self.dim1 = hyper_param['dim1']
        self.dim2 = hyper_param['dim2']
        self.model_size = hyper_param['model_size']
        self.dropout = hyper_param['dropout']
        self.weight_decay = hyper_param['weight_decay']
        self.print_every = hyper_param['print_every']
        self.val_every = hyper_param['val_every']
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
        self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)
        self.W1 = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=False)

        self.layers = nn.ModuleList()
        input_dim = self.embedding_size
        for layer in range(num_layers):
            aggregator = GAT(input_dim, input_dim, act=self.act, dropout=self.dropout, batch_norm=True ,model_size=self.model_size)
            self.layers.append(aggregator)

        self.W2 = nn.Linear(input_dim + self.embedding_size, self.embedding_size, bias=False)

    def individual_interest(self, input_session):
        input = torch.LongTensor(input_session[0])  # input.shape : [max_length]
        emb_seqs = self.item_embedding(input)  # emb_seqs.shape : [max_length, embedding_dim]
        emb_seqs = torch.unsqueeze(emb_seqs, 0)

        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)

        for batch in range(self.batch_size - 1):
            input = torch.LongTensor(input_session[batch + 1])
            emb_seq = self.item_embedding(input)
            emb_seq = torch.unsqueeze(emb_seq, 0)
            emb_seqs = torch.cat((emb_seqs, emb_seq), 0)

        hu, (_, _) = self.lstm(emb_seqs)  # output.shape : [batch_size, max_length, embedding_dim]

        return hu

    def friends_interest(self, support_nodes_layer1, support_nodes_layer2, support_sessions_layer1, support_sessions_layer2):
        long_term = []
        support_nodes_layer = [support_nodes_layer1, support_nodes_layer2]

        for layer in support_nodes_layer:
            long_input = torch.LongTensor(layer)
            long_term1_2 = self.user_embedding(long_input)
            long_term.append(long_term1_2)
            # long_term[0].shape : [sample1 * sample2, embedding_dim]
            # long_term[1].shape : [sample2, embedding_dim]

        short_term = []
        support_sessions_layer = [support_sessions_layer1, support_sessions_layer2]
        sample1_2 = [self.samples_1 * self.samples_2, self.samples_2]

        for layer, sample in zip(support_sessions_layer, sample1_2):
            short_arange = torch.arange(self.batch_size * sample, dtype=torch.long)
            short_input = torch.LongTensor(layer)[short_arange]
            friend_emb_seqs = self.item_embedding(short_input)

            if self.feat_drop is not None:
                friend_emb_seqs = self.feat_drop(friend_emb_seqs)

            short_term1_2, (_, _) = self.lstm(friend_emb_seqs)

            short_term1_2 = short_term1_2[:, 0, :]  # [, max_length, ] -> [, ]
            short_term.append(short_term1_2)
            # short_term[0].shape : [batch_size * sample1 * sample2, embedding_dim]
            # short_term[1].shape : [batch_size * sample2, embedding_dim]

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

    def score(self, feed_dict):
        hu = self.individual_interest(feed_dict['input_session'])

        long_short_term = self.friends_interest(feed_dict['support_nodes_layer1'],
                                                feed_dict['support_nodes_layer2'],
                                                feed_dict['support_sessions_layer1'],
                                                feed_dict['support_sessions_layer2'])

        # GAT
        hu = torch.swapaxes(hu, 0, 1)
        outputs = []
        next_hidden = []
        support_sizes = [1, self.samples_2, self.samples_1 * self.samples_2]
        num_samples = [self.samples_1, self.samples_2]
        for i in range(self.max_length):
            hu_ = hu[i]  # implement 1 of 20
            for layer in self.layers:
                hidden = [hu_, long_short_term[0], long_short_term[1]]
                for hop in range(self.num_layers):
                    neigh_dims = [self.batch_size * support_sizes[hop],
                                  num_samples[self.num_layers - hop - 1],
                                  100]
                    h = layer([hidden[hop],
                               torch.reshape(hidden[hop + 1], neigh_dims)])
                    next_hidden.append(h)
            outputs.append(next_hidden[0])
        feat = torch.stack(outputs, axis=0)
        # hu.shape, feat.shape : [max_length, batch, embedding_size]

        sr = self.W2(torch.cat((hu, feat), dim=2))  # final representation

        logits = sr @ self.item_embedding(self.item_indices).t()  # prediction
        # logit shape : [max_length, batch, item_embedding]

        mask = torch.LongTensor(feed_dict['mask_y'])
        logits = torch.swapaxes(logits, 0, 1)
        logits *= torch.unsqueeze(mask, 2)

        # return : [batch, max_length, item_embedding]
        return logits

    def forward(self, feed_dict, labels):
        logits = self.score(feed_dict)

        logits = torch.swapaxes(logits, 1, 2)
        logits = logits.to(dtype=torch.float)
        # logits : [batch, item_embedding, max_length]
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        # labels : [batch, max_length]

        loss = F.cross_entropy(logits, labels)

        return loss

    def predict(self, feed_dict):
        logits = self.score(feed_dict)
        return logits
