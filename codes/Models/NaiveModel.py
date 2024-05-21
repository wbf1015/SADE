import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class NaiveModel(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, entity_pruner=None, relation_pruner=None, loss=None, args=None):
        super(NaiveModel, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.EntityPruner = entity_pruner
        self.RelationPruner = relation_pruner
        self.loss = loss
        self.args = args
    
    def get_postive_score(self, score):
        return score[:, 0]
    
    def get_negative_score(self, score):
        return score[:, 1:]
    
    def forward(self, data, subsampling_weight, mode):
        head, relation, tail = self.EmbeddingManager(data, mode)
        if self.args.cuda:
            head, relation, tail = head.cuda(), relation.cuda(), tail.cuda()
        head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), self.EntityPruner(tail)
        score = self.KGE(head, relation, tail, mode, self.args)
        p_score, n_score = self.get_postive_score(score), self.get_negative_score(score)
        loss, loss_record = self.loss(p_score, n_score, subsampling_weight)
        return loss, loss_record
    
    def predict(self, data, mode):
        head, relation, tail = self.EmbeddingManager(data, mode)
        head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), self.EntityPruner(tail)
        score = self.KGE(head, relation, tail, mode, self.args)
        return score
        
        