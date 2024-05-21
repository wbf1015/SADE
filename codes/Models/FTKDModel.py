import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class FTKDModel(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, entity_pruner=None, relation_pruner=None, loss=None, kdloss=None, args=None, FineTuner=None):
        super(FTKDModel, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.EntityPruner = entity_pruner
        self.RelationPruner = relation_pruner
        self.loss = loss
        self.KDLoss = kdloss
        self.FineTuner = FineTuner
        self.args = args
        self.KDLossWeight = args.kdloss_weight
        self.dict = None
        
    def get_postive_score(self, score):
        return score[:, 0]
    
    def get_negative_score(self, score):
        return score[:, 1:]
    
    def forward(self, data, subsampling_weight, mode):
        origin_head, origin_relation, origin_tail, head, relation, tail = self.EmbeddingManager(data, mode)
        if self.args.cuda:
            origin_head, origin_relation, origin_tail, head, relation, tail = origin_head.cuda(), origin_relation.cuda(), origin_tail.cuda(), head.cuda(), relation.cuda(), tail.cuda()
        
        t_score = self.KGE(origin_head, origin_relation, origin_tail, mode, self.args)
        
        head_, relation, tail_ = self.EntityPruner(origin_head), self.RelationPruner(relation), self.EntityPruner(origin_tail)
        
        head, tail = self.FineTuner(head_, head), self.FineTuner(tail_, tail)
        score = self.KGE(head, relation, tail, mode, self.args)
        p_score, n_score = self.get_postive_score(score), self.get_negative_score(score)
        loss, loss_record = self.loss(p_score, n_score, subsampling_weight)
        KDloss, KDloss_record = self.KDLoss(t_score, score, subsampling_weight)
        
        loss = loss + self.KDLossWeight * KDloss
        loss_record.update(KDloss_record)
        loss_record.update({'total_loss':loss.item()})
        
        self.dict={
            'data' : data,
            'mode' : mode,
            'head' : head,
            'tail' : tail,
        }
        
        # self.EmbeddingManager.update_embedding(data, mode, head, tail)
        return loss, loss_record
    
    def predict(self, data, mode):
        origin_head, origin_relation, origin_tail, head, relation, tail = self.EmbeddingManager(data, mode)
        if self.args.cuda:
            origin_head, origin_relation, origin_tail, head, relation, tail = origin_head.cuda(), origin_relation.cuda(), origin_tail.cuda(), head.cuda(), relation.cuda(), tail.cuda()
        score = self.KGE(head, relation, tail, mode, self.args)
        return score
        
    def set_kdloss(self, kdloss):
        self.KDLoss = kdloss
    
    def update_embedding(self):
        self.EmbeddingManager.update_embedding(self.dict['data'], self.dict['mode'], self.dict['head'], self.dict['tail'])