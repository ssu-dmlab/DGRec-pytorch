#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import sys
import fire
import torch
from pathlib import Path

from utils import set_random_seed
from data import MyDataset
from models.DGRec.train import MyTrainer
from models.DGRec.eval import MyEvaluator
from utils import log_param
from utils import set_graph
from loguru import logger


def run_mymodel(device, data, hyper_param):
    g = set_graph(data[0])

    trainer = MyTrainer(device=device)

    model = trainer.train_with_hyper_param(data=data,
                                           hyper_param=hyper_param, graph=g)

    evaluator = MyEvaluator(device=device)
    accuracy, real_accuracy = evaluator.evaluate(model, data, hyper_param, g)

    return accuracy, real_accuracy


def main(model='DGRec',
         seed = 123,
         training=True,
         global_only = False,
         local_only = False,
         epochs = 20,
         aggregator_type = 'attn',
         act = 'relu',
         batch_size = 200,
         max_degree = 50,
         num_users = -1,
         num_items = 100,
         concat = False,
         learning_rate = 0.001,
         hidden_size = 100,
         embedding_size = 50,
         emb_user = 50,
         max_length = 20,
         samples_1 = 10,
         samples_2 = 5,
         dim1 = 100,
         dim2 = 100,
         model_size = 'small',
         dropout = 0.,
         weight_decay = 0.,
         print_every = 100,
         val_every = 500,
         ckpt_dir = 'save/'):
    """
    Handle user arguments of ml-project-template

    :param model: name of model to be trained and tested
    :param seed: random_seed (if -1, a default seed is used)
    :param batch_size: size of batch
    :param epochs: number of training epochs
    :param learning_rate: learning rate
    """

    # Step 0. Initialization
    logger.info("The main procedure has started with the following parameters:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_random_seed(seed=seed, device=device)
    param = dict()
    param['model'] = model
    param['seed'] = seed
    param['training'] = training
    param['global_only'] = global_only
    param['local_only'] = local_only
    param['concat'] = concat
    param['model_size'] = model_size
    param['ckpt_dir'] = ckpt_dir
    param['device'] = device
    log_param(param)

    # Step 1. Load datasets
    data_path = '/Users/kimtaesu/PycharmProjects/DGRec-pytorch/datasets/musicdata/'
    #logger.info("path of data is:{}".format(data_path))
    MyData = MyDataset(data_path)
    data = MyData.load_data()
    adj_info = data[0]
    latest_per_user_by_time = data[1]
    user_id_map = data[2]
    item_id_map = data[3]
    train_df = data[4]
    valid_df = data[5]
    test_df = data[6]

    logger.info("The datasets are loaded where their statistics are as follows:")
    logger.info("- # of training instances: {}".format(len(train_df)))
    logger.info("- # of test instances: {}".format(len(test_df)))
    # Step 2. Run (train and evaluate) the specified model

    logger.info("Training the model has begun with the following hyperparameters:")
    num_items = len(item_id_map)
    num_users = len(user_id_map)

    hyper_param = dict()
    hyper_param['epochs'] = epochs
    hyper_param['aggregator_type'] = aggregator_type
    hyper_param['act'] = act
    hyper_param['batch_size'] = batch_size
    hyper_param['max_degree'] = max_degree
    hyper_param['num_users'] = num_users
    hyper_param['num_items'] = num_items
    hyper_param['learning_rate'] = learning_rate
    hyper_param['hidden_size'] = hidden_size
    hyper_param['embedding_size'] = embedding_size
    hyper_param['emb_user'] = emb_user
    hyper_param['max_length'] = max_length
    hyper_param['samples_1'] = samples_1
    hyper_param['samples_2'] = samples_2
    hyper_param['dim1'] = dim1
    hyper_param['dim2'] = dim2
    hyper_param['dropout'] = dropout
    hyper_param['weight_decay'] = weight_decay
    hyper_param['print_every'] = print_every
    hyper_param['val_every'] = val_every
    log_param(hyper_param)


    if model == 'DGRec':
        accuracy, real_accuracy = run_mymodel(device=device,
                               data=data,
                               hyper_param=hyper_param)

        # - If you want to add other model, then add an 'elif' statement with a new runnable function
        #   such as 'run_my_model' to the below
        # - If models' hyperparamters are varied, need to implement a function loading a configuration file
    else:
        logger.error("The given \"{}\" is not supported...".format(model))
        return

    # Step 3. Report and save the final results
    logger.info("The model has been trained. The test accuracy is {:.4} and real_accuracy is {:.4}.".format(accuracy, real_accuracy))


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
