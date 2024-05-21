import argparse
import json
import logging
import os
import random
import copy

import numpy as np
import torch

from torch.utils.data import DataLoader


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Pruning Models',
        usage='Run.py [<args>] [-h | --help]'
    )

    parser.add_argument('-cuda', '--cuda', action='store_true', help='use GPU')
    parser.add_argument('-seed', '--seed', default=42, type=int, help='manual_set_random_seed')
    parser.add_argument('-data_path', '--data_path', type=str, default=None)
    parser.add_argument('-entity_mul', '--entity_mul', type=int, default=1)
    parser.add_argument('-relation_mul', '--relation_mul', type=int, default=1)
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-batch_size', '--batch_size', default=1024, type=int)
    parser.add_argument('-negative_sample_size', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-steps','--steps', default=100000, type=int)
    parser.add_argument('-hidden_dim', '--hidden_dim', default=512, type=int)
    parser.add_argument('-target_dim', '--target_dim', default=None, type=int, help='feature pruning with method')
    parser.add_argument('-gamma', '--gamma', default=12.0, type=float)
    parser.add_argument('-dropout', '--dropout', default=0.0, type=float)
    parser.add_argument('-negative_adversarial_sampling', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-adversarial_temperature', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-lamma', '--lamma', default=0.01, type=float)
    parser.add_argument('-regularization', '--regularization', default=0.0, type=float)
    
    parser.add_argument('-test_batch_size', '--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    
    parser.add_argument('-optimizer', '--optimizer', default='SGD', type=str)
    parser.add_argument('-scheduler', '--scheduler', default='MultiStepLR', type=str)
    parser.add_argument('-momentum', '--momentum', default=0.0, type=float)
    parser.add_argument('-weight_decay', '--weight_decay', default=0.0, type=float)
    parser.add_argument('-patience', '--patience', default=2000, type=int)
    parser.add_argument('-cooldown', '--cooldown', default=2000, type=int)
    parser.add_argument('-warm_up_steps', '--warm_up_steps', default=None, type=int)
    parser.add_argument('-decreasing_lr', '--decreasing_lr', default=0.1, type=float)
    
    parser.add_argument('-init_checkpoint', '--init_checkpoint', default='without', type=str)
    parser.add_argument('-save_path', '--save_path', default=None, type=str)
    parser.add_argument('-pretrain_path', '--pretrain_path', type=str, default='without', help='pretrained model path')
    parser.add_argument('-save_checkpoint_steps', '--save_checkpoint_steps', default=1000, type=int)
    parser.add_argument('-log_steps', '--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('-test_per_steps', '--test_per_steps', default=10000, type=int, help='test every xx steps')
    parser.add_argument('-test_log_steps', '--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('-nentity', '--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('-nrelation', '--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('-t_dff', '--t_dff', type=int, default=4, help='Transformer dff dimension')
    parser.add_argument('-t_layer', '--t_layer', type=int, default=1, help='Transformer layer num')
    parser.add_argument('-head1', '--head1', type=int, default=32, help='Transformer HEAD1')
    parser.add_argument('-head2', '--head2', type=int, default=32, help='Transformer HEAD2')
    parser.add_argument('-head3', '--head3', type=int, default=32, help='Transformer HEAD3')
    parser.add_argument('-head4', '--head4', type=int, default=32, help='Transformer HEAD4')
    
    parser.add_argument('-token1', '--token1', type=int, default=4, help='token nums in TokenAttention')
    parser.add_argument('-token2', '--token2', type=int, default=4, help='token nums in TokenAttention')
    
    parser.add_argument('-kd_gamma', '--kd_gamma', default=12.0, type=float)
    parser.add_argument('-kdloss_weight', '--kdloss_weight', type=float, default=0.01, help='KD loss weight')
    
    parser.add_argument('-temperature', '--temperature', type=float, default=0.1, help='contrastive gamma')
    parser.add_argument('-contrastive_gamma', '--contrastive_gamma', type=float, default=12.0, help='contrastive gamma')
    parser.add_argument('-ckdloss_weight', '--ckdloss_weight', type=float, default=0.001, help='Contrastive loss weight')
    parser.add_argument('-ckdloss_dropout', '--ckdloss_dropout', type=float, default=0.05, help='Contrastive construct dropout')
    return parser.parse_args(args)


def set_logger(args):
    log_file = os.path.join(args.save_path, 'train.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def read_data(args):
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    logging.info('#total entity: %d' % nentity)
    logging.info('#total relation: %d' % nrelation)
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    return train_triples, valid_triples, test_triples, all_true_triples, nentity, nrelation