#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def load_adj(self, data_path):
        df_adj = pd.read_csv(data_path + '/adj.tsv', sep='\t', dtype={0: np.int32, 1: np.int32})
        return df_adj

    def load_latest_session(self, data_path):
        ret = []
        for line in open(data_path + '/latest_sessions.txt'):
            chunks = line.strip().split(',')
            ret.append(chunks)
        return ret

    def load_map(self, data_path, name='user'):
        if name == 'user':
            file_path = data_path + '/user_id_map.tsv'
        elif name == 'item':
            file_path = data_path + '/item_id_map.tsv'
        else:
            raise NotImplementedError
        id_map = {}
        for line in open(file_path):
            k, v = line.strip().split('\t')
            id_map[k] = str(v)
        return id_map

    def load_data(self):
        adj = self.load_adj(self.data_path)
        latest_sessions = self.load_latest_session(self.data_path)
        user_id_map = self.load_map(self.data_path, 'user')
        item_id_map = self.load_map(self.data_path, 'item')
        train = pd.read_csv(self.data_path + '/train.tsv', sep='\t', dtype={0: np.int32, 1: np.int32, 3: np.float32})
        valid = pd.read_csv(self.data_path + '/valid.tsv', sep='\t', dtype={0: np.int32, 1: np.int32, 3: np.float32})
        test = pd.read_csv(self.data_path + '/test.tsv', sep='\t', dtype={0: np.int32, 1: np.int32, 3: np.float32})
        return [adj, latest_sessions, user_id_map, item_id_map, train, valid, test]
