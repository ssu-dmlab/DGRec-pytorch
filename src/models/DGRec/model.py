#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import glorot, zeros


class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, neigh_input_dim=None, dropout=0., bias=False, act=nn.ReLU(), concat=False, **kwargs):
        super().__init__()

        self.dropout = dropout
        self.bias = bias
        self.act = nn.ReLU()
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        self.vars = {}
        self.vars['weights'] = glorot([neigh_input_dim, output_dim])

        if self.bias:
            self.vars['bias'] = zeros([self.output_dim])

        self.fc = nn.Linear(100, 100, bias=True)
        self.m = nn.Softmax(dim=-1)

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, inputs):
        self_vecs, neigh_vecs = inputs

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
        self.user_embedding = nn.Embedding(self.num_users,
                                           self.embedding_size)  # (num_users=26511, embedding_dim=100)
        self.item_embeeding = nn.Embedding(self.num_items,
                                           self.embedding_size,
                                           padding_idx=0)  # (num_items=12592, embedding_dim=100)
        self.item_indices = nn.Parameter(torch.arange(0, self.num_items, dtype=torch.long),
                                         requires_grad=False)
        self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)
        self.W1 = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=False)

        self.layers = nn.ModuleList()
        input_dim = self.embedding_size
        for layer in range(num_layers):
            aggregator = GAT(self.hidden_size, self.hidden_size, act=self.act,
                            dropout=self.dropout if self.training else 0., model_size=self.model_size)

            self.layers.append(aggregator)

        self.W2 = nn.Linear(input_dim + self.embedding_size, self.embedding_size, bias=False)

    def score(self, feed_dict):
        # Part-1 : Individual Interest
        input = torch.LongTensor(feed_dict['input_session'][0])  # input.shape : [max_length]
        emb_seqs = self.item_embeeding(input)  # emb_seqs.shape : [max_length, embedding_dim]
        emb_seqs = torch.unsqueeze(emb_seqs, 0)
        for batch in range(self.batch_size - 1):
            input = torch.LongTensor(feed_dict['input_session'][batch + 1])
            emb_seq = self.item_embeeding(input)
            emb_seq = torch.unsqueeze(emb_seq, 0)
            emb_seqs = torch.cat((emb_seqs, emb_seq), 0)
        # emb_seqs = emb_seqs.view(self.batch_size, self.max_length, self.embedding_size)

        hu, (_, _) = self.lstm(emb_seqs)  # output.shape : [batch_size, max_length, embedding_dim]

        # Part-2 : Friends' Interest
        # long-term
        long_input1 = torch.LongTensor(feed_dict['support_nodes_layer1'])
        long_input2 = torch.LongTensor(feed_dict['support_nodes_layer2'])

        long_term1 = self.user_embedding(long_input1)  # long_term1.shape : [sample1 * sample2, embedding_dim]
        long_term2 = self.user_embedding(long_input2)  # long_term2.shape : [sample2, embedding_dim]

        long_term = [long_term2, long_term1]

        # short-term
        short_input1 = torch.LongTensor(feed_dict['support_sessions_layer1'][0])  # input.shape : [max_length]
        friend_emb_seqs1 = self.item_embeeding(short_input1)  # emb_seqs.shape : [max_length, embedding_dim]
        for batch in range(self.batch_size * self.samples_1 * self.samples_2 - 1):
            short_input1 = torch.LongTensor(feed_dict['support_sessions_layer1'][batch + 1])
            friend_emb_seq1 = self.item_embeeding(short_input1)
            friend_emb_seqs1 = torch.cat((friend_emb_seqs1, friend_emb_seq1), 0)
        friend_emb_seqs1 = friend_emb_seqs1.view(self.batch_size * self.samples_1 * self.samples_2, self.max_length,
                                                 self.embedding_size)

        short_term1, (_, _) = self.lstm(friend_emb_seqs1)
        # short_term1.shape : [batch_size * sample1 * sample2, max_length, embedding_dim]

        short_term1 = short_term1[:, 0, :]  # [, max_length, ] -> [, ]

        short_input2 = torch.LongTensor(feed_dict['support_sessions_layer2'][0])  # input.shape : [max_length]
        friend_emb_seqs2 = self.item_embeeding(short_input2)  # emb_seqs.shape : [max_length, embedding_dim]
        for batch in range(self.batch_size * self.samples_2 - 1):
            short_input2 = torch.LongTensor(feed_dict['support_sessions_layer2'][batch + 1])
            friend_emb_seq2 = self.item_embeeding(short_input2)
            friend_emb_seqs2 = torch.cat((friend_emb_seqs2, friend_emb_seq2), 0)
        friend_emb_seqs2 = friend_emb_seqs2.view(self.batch_size * self.samples_2, self.max_length, self.embedding_size)

        short_term2, (_, _) = self.lstm(friend_emb_seqs2)
        # short_term2.shape : [batch_size * sample2, max_length, embedding_dim]

        short_term2 = short_term2[:, 0, :]  # [, max_length, ] -> [, ]

        short_term = [short_term2, short_term1]

        # long-term & short-term
        long_short_term1 = torch.cat((long_term[1], short_term[1]), dim=1)
        long_short_term2 = torch.cat((long_term[0], short_term[0]), dim=1)
        # long_short_term1.shape : [batch_size * sample1 * sample2, embedding_dim + embedding_dim]
        # long_short_term2.shape : [batch_size * sample2, embedding_dim + embedding_dim]

        long_short_term1 = torch.relu(self.W1(long_short_term1))
        long_short_term2 = torch.relu(self.W1(long_short_term2))
        # long_short_term1.shape : [batch_size * sample1 * sample2, embedding_dim]
        # long_short_term2.shape : [batch_size * sample2, embedding_dim]

        long_short_term = [long_short_term2, long_short_term1]

        # Part-3 : Graph-Attention Network
        hu = torch.swapaxes(hu, 0, 1)
        outputs = []
        next_hidden = []
        support_sizes = [1, self.samples_2, self.samples_1 * self.samples_2]
        num_samples = [self.samples_1, self.samples_2]
        for i in range(self.max_length):
            for layer in self.layers:
                hu_ = hu[i]  # implement 1 of 20
                hidden = [hu_, long_short_term[0], long_short_term[1]]
                for hop in range(self.num_layers):
                    neigh_dims = [self.batch_size * support_sizes[hop],
                                  num_samples[len(num_samples) - hop - 1],
                                  100]
                    h = layer([hidden[hop],
                               torch.reshape(hidden[hop + 1], neigh_dims)])
                    next_hidden.append(h)
            outputs.append(next_hidden[0])
        feat = torch.stack(outputs, axis=0)
        # hu.shape, feat : [max_length, batch, embedding_size]

        sr = self.W2(torch.cat((hu, feat), dim=2))  # final representation

        logits = sr @ self.item_embeeding(self.item_indices).t()  # prediction
        # logit shape : [max_length, batch, item_embedding]

        mask = torch.LongTensor(feed_dict['mask_y'])
        logits = torch.swapaxes(logits, 0, 1)
        logits *= torch.unsqueeze(mask, 2)

        return logits

    def forward(self, feed_dict, labels):
        logits = self.score(feed_dict)

        logits = torch.swapaxes(logits, 1, 2)
        logits = logits.to(dtype=torch.float)
        labels = torch.tensor(np.array(labels), dtype=torch.long)

        loss = F.cross_entropy(logits, labels) # logits : [batch, item_embedding, max_length] / labels : [batch, max_length]

        return loss

    def predict(self, feed_dict):
        logits = self.score(feed_dict)
        return torch.argmax(logits, dim=2)
