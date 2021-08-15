#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from models.DGRec.model import DGRec
from models.DGRec.batch.minibatch import MinibatchIterator
from tqdm import tqdm


class MyEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, data, hyper_param, graph):
        with torch.no_grad():
            adj_info = data[0]
            latest_per_user_by_time = data[1]
            user_id_map = data[2]
            item_id_map = data[3]
            train_df = data[4]
            valid_df = data[5]
            test_df = data[6]

            epochs = hyper_param['epochs']
            aggregator_type = hyper_param['aggregator_type']
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
            dropout = hyper_param['dropout']
            weight_decay = hyper_param['weight_decay']
            print_every = hyper_param['print_every']
            val_every = hyper_param['val_every']


            minibatch = MinibatchIterator(adj_info,
                                          latest_per_user_by_time,
                                          [train_df, valid_df, test_df],
                                          batch_size=batch_size,
                                          max_degree=max_degree,
                                          num_nodes=len(user_id_map),
                                          max_length=max_length,
                                          samples_1_2=[samples_1, samples_2],
                                          training=False)

            model.eval()

            feed_dict = minibatch.next_val_minibatch_feed_dict("test")

            labels = torch.tensor(np.array(feed_dict['output_session']), dtype=torch.long)

            predictions = model.predict(feed_dict, graph)

            print(labels)
            print(predictions)

            corrects = predictions == labels
            accuracy = corrects.float().mean()

            real_corrects = ((predictions == labels) * (labels != 0))
            real_accuracy = real_corrects.float().mean()

        return accuracy.item(), real_accuracy.item()
