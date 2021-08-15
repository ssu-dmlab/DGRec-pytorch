#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import dgl
from loguru import logger
from data import MyDataset


def set_random_seed(seed, device):
    # for reproducibility (always not guaranteed in pytorch)
    # [1] https://pytorch.org/docs/stable/notes/randomness.html
    # [2] https://hoya012.github.io/blog/reproducible_pytorch/

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def log_param(param):
    for key, value in param.items():
        if type(value) is dict:
            for in_key, in_value in value.items():
                logger.info('{:20}:{:>50}'.format(in_key, '{}'.format(in_value)))
        else:
            logger.info('{:20}:{:>50}'.format(key, '{}'.format(value)))

def set_graph(adj):
    data = (torch.LongTensor(adj['Follower']), torch.LongTensor(adj['Follower']))
    g = dgl.graph(data)

    g.ndata['x'] = torch.ones(g.num_nodes(), 32)
    g.edata['weights'] = torch.rand(g.num_edges())

    return g

if __name__ == "__main__":
    data_path = '/Users/kimtaesu/PycharmProjects/DGRec-pytorch/datasets'
    #logger.info("path of data is:{}".format(data_path))
    MyData = MyDataset(data_path)
    data = MyData.load_data()
    print(set_graph(data[0]))
