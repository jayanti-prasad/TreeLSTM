from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import Constants
from model import SimilarityTreeLSTM
from vocab import Vocab
from dataset import SICKDataset
from metrics import Metrics
import utils
from trainer  import Trainer
import  config 
import argparse 
import configparser 

from data_util import DataUtil, get_embd 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cmod')
    parser.add_argument('-c', '--config', help='Config file path', required=True)
    cfg_parser = configparser.ConfigParser()

    args = parser.parse_args()
    cfg_parser.read(args.config)

    cfg = config.Config(cfg_parser)

    D = DataUtil (cfg)

    train_dataset = D.get_data('train')
    test_dataset = D.get_data('test')
    dev_dataset = D.get_data('dev')

    device = torch.device("cuda:0" if cfg.use_cuda() else "cpu")

    if cfg.sparse() and cfg.weight_decay() != 0:
        cfg.logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()

    torch.manual_seed(cfg.random_seed())
    random.seed(cfg.random_seed())

    if cfg.use_cuda():
        torch.cuda.manual_seed(cfg.random_seed())
        torch.backends.cudnn.benchmark = True


    cfg.logger.info('==> SICK vocabulary size : %d ' % D.vocab.size())
    cfg.logger.info('==> Size of train data   : %d ' % len(train_dataset))
    cfg.logger.info('==> Size of dev data     : %d ' % len(dev_dataset))
    cfg.logger.info('==> Size of test data    : %d ' % len(test_dataset))

    model = SimilarityTreeLSTM(
        D.vocab.size(),
        cfg.input_dim(),
        cfg.mem_dim(),
        cfg.hidden_dim(),
        cfg.num_classes(),
        cfg.sparse(),
        cfg.freeze_embed())

    criterion = nn.KLDivLoss()

    cfg.logger.info("model:\n" + str(model))

    emb = get_embd(cfg, D.vocab)

    # plug these into embedding matrix inside model
    model.emb.weight.data.copy_(emb)

    model.to(cfg.device()), criterion.to(cfg.device())

    metrics = Metrics(cfg.num_classes())

    # create trainer object for training and testing


    trainer = Trainer(cfg, model, criterion, cfg.optimizer(model), cfg.device())

    best = -float('inf')
    for epoch in range(cfg.num_epochs()):
        train_loss = trainer.train(train_dataset)

        train_loss, train_pred = trainer.test(train_dataset)
        test_loss, test_pred = trainer.test(test_dataset)

        train_pearson = metrics.pearson(train_pred, train_dataset.labels)
        train_mse = metrics.mse(train_pred, train_dataset.labels)
        cfg.logger.info('==> Epoch {}, Train \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, train_loss, train_pearson, train_mse))

        test_pearson = metrics.pearson(test_pred, test_dataset.labels)
        test_mse = metrics.mse(test_pred, test_dataset.labels)
        cfg.logger.info('==> Epoch {}, Test \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, test_loss, test_pearson, test_mse))

        if best < test_pearson:
            best = test_pearson
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'pearson': test_pearson, 'mse': test_mse,
                'args': args, 'epoch': epoch
            }
            cfg.logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(cfg.output_dir(), cfg.model_name()))
       
