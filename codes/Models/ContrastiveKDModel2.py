import sys
import os
import logging

import torch.nn.functional as F
import torch
import torch.nn as nn

class ContrastiveKDModel2(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, entity_pruner=None, relation_pruner=None, loss=None, KDloss=None, ContrastiveLoss=None, args=None):
        super(ContrastiveKDModel2, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.EntityPruner = entity_pruner
        self.RelationPruner = relation_pruner
        self.loss = loss
        self.KDLoss = KDloss
        self.KDLossWeight = args.kdloss_weight
        self.ContrastiveLossWeight = args.ckdloss_weight
        self.args = args
    
    def regulation_loss(self, head, relation, tail):
        head_norm = head.norm(p=3)**3
        relation_norm = self.EmbeddingManager.relation_embedding.norm(p=3).norm(p=3)**3
        tail_norm = tail.norm(p=3)**3

        batch, neg_sampling = head.shape[0], max(head.shape[1], tail.shape[1]) + 1
        bias = pow((batch*neg_sampling)/self.args.nentity, 1/3)
        
        regularization = ((head_norm + tail_norm) / (bias)) + relation_norm
        regularization = regularization * self.args.regularization
        
        return regularization, {'regularization_loss': regularization.item()}
    
    def regulation_loss2(self, head, relation, tail):
        head, tail = head.view(-1, head.shape[2]), tail.view(-1, tail.shape[2])
        entity = torch.cat((head, tail), dim=0)
        indices = torch.randperm(entity.size(0))
        
        entity = entity[indices]
        entity = entity[:self.args.nentity]
        
        entity_norm = entity.norm(p=3)**3
        relation_norm = self.EmbeddingManager.relation_embedding.norm(p=3).norm(p=3)**3
        
        regularization = entity_norm + relation_norm
        regularization = regularization * self.args.regularization
        
        return regularization, {'regularization_loss': regularization.item()}

    
    def get_postive_score(self, score):
        return score[:, 0]
    
    def get_negative_score(self, score):
        return score[:, 1:]
    
    def forward(self, data, subsampling_weight, mode):
        head, relation, tail, origin_relation = self.EmbeddingManager(data, mode)
        if self.args.cuda:
            head, relation, tail = head.cuda(), relation.cuda(), tail.cuda()
        t_score = self.KGE(head, origin_relation, tail, mode, self.args)

        if mode == 'head-batch':
            relation, tail = self.RelationPruner(relation), self.EntityPruner(tail)
            head, closs, closs_record = self.EntityPruner(head, True, subsampling_weight)
        elif mode == 'tail-batch':
            head, relation = self.EntityPruner(head), self.RelationPruner(relation)
            tail, closs, closs_record = self.EntityPruner(tail, True, subsampling_weight)
         
        score = self.KGE(head, relation, tail, mode, self.args)
        p_score, n_score = self.get_postive_score(score), self.get_negative_score(score)
        loss, loss_record = self.loss(p_score, n_score, subsampling_weight)
        KDloss, KDloss_record = self.KDLoss(t_score, score, subsampling_weight)
        
        # if type(self.KGE).__name__ == 'ComplEx':
        #     self.EmbeddingManager.update_embedding(data, mode, head, tail)
        
        if self.args.regularization != 0.0:
           rloss, rloss_record = self.regulation_loss2(head, relation, tail)
           loss = loss + rloss
           loss_record.update(rloss_record)
          
        loss = loss + self.KDLossWeight * KDloss + self.ContrastiveLossWeight * closs
        loss_record.update(KDloss_record)
        loss_record.update(closs_record)
        loss_record.update({'total_loss':loss.item()})
        return loss, loss_record
    
    def predict(self, data, mode):
        head, relation, tail, _ = self.EmbeddingManager(data, mode)
        head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), self.EntityPruner(tail)
        score = self.KGE(head, relation, tail, mode, self.args)
        return score

    def set_kdloss(self,kdloss):
        self.KDLoss = kdloss
        
        