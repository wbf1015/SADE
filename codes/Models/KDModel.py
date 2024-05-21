import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class KDModel(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, entity_pruner=None, relation_pruner=None, loss=None, KDloss=None, args=None):
        super(KDModel, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.EntityPruner = entity_pruner
        self.RelationPruner = relation_pruner
        self.loss = loss
        self.KDLoss = KDloss
        self.KDLossWeight = args.kdloss_weight
        self.args = args
    
    def get_postive_score(self, score):
        return score[:, 0]
    
    def get_negative_score(self, score):
        return score[:, 1:]
    
    def forward(self, data, subsampling_weight, mode):
        head, relation, tail, origin_relation = self.EmbeddingManager(data, mode)
        if self.args.cuda:
            head, relation, tail = head.cuda(), relation.cuda(), tail.cuda()
        t_score = self.KGE(head, origin_relation, tail, mode, self.args)
        head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), self.EntityPruner(tail)
        score = self.KGE(head, relation, tail, mode, self.args)
        p_score, n_score = self.get_postive_score(score), self.get_negative_score(score)
        loss, loss_record = self.loss(p_score, n_score, subsampling_weight)
        KDloss, KDloss_record = self.KDLoss(t_score, score, subsampling_weight)
        
        loss = loss + self.KDLossWeight * KDloss
        loss_record.update(KDloss_record)
        loss_record.update({'total_loss':loss.item()})
        return loss, loss_record
    
    def predict(self, data, mode):
        head, relation, tail, _ = self.EmbeddingManager(data, mode)
        head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), self.EntityPruner(tail)
        score = self.KGE(head, relation, tail, mode, self.args)
        return score

    def set_kdloss(self,kdloss):
        self.KDLoss = kdloss
        
        