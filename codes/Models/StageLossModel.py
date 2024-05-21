import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class StageLossModel(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, entity_pruner=None, relation_pruner=None, loss=None, args=None):
        super(StageLossModel, self).__init__()
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
            
        head, tmp_head = self.EntityPruner(head)
        relation, tmp_relation = self.RelationPruner(relation)
        tail, tmp_tail = self.EntityPruner(tail)
        
        score = self.KGE(head, relation, tail, mode)
        p_score, n_score = self.get_postive_score(score), self.get_negative_score(score)
        
        tmp_score = self.KGE(tmp_head, tmp_relation, tmp_tail, mode, real_dim=tmp_relation.shape[-1])
        tmp_p_score, tmp_n_score = self.get_postive_score(tmp_score), self.get_negative_score(tmp_score) 
        
        loss, loss_record = self.loss(p_score, n_score, subsampling_weight)
        tmp_loss, tmp_loss_record = self.loss(tmp_p_score, tmp_n_score, subsampling_weight)
        modified_tmp_loss_record = {'tmp' + key: value for key, value in tmp_loss_record.items()}
        
        loss = (loss+tmp_loss)//2
        loss_record.update(modified_tmp_loss_record)
        
        return loss, loss_record
    
    def predict(self, data, mode):
        head, relation, tail = self.EmbeddingManager(data, mode)
        head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), self.EntityPruner(tail)
        score = self.KGE(head, relation, tail, mode)
        return score
        
        