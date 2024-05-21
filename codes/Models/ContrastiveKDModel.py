import sys
import os
import logging

import torch.nn.functional as F
import torch
import torch.nn as nn

class ContrastiveKDModel(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, entity_pruner=None, relation_pruner=None, loss=None, KDloss=None, ContrastiveLoss=None, args=None):
        super(ContrastiveKDModel, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.EntityPruner = entity_pruner
        self.RelationPruner = relation_pruner
        self.loss = loss
        self.KDLoss = KDloss
        self.ContrastiveLoss = ContrastiveLoss
        self.KDLossWeight = args.kdloss_weight
        self.ContrastiveLossWeight = args.ckdloss_weight
        self.args = args
    
    def get_postive_score(self, score):
        return score[:, 0]
    
    def get_negative_score(self, score):
        return score[:, 2:]
    
    def delete_s_score(self, s_score):
        return torch.cat((s_score[:, :1], s_score[:, 2:]), dim=1)
    
    def make_contrastive_pair(self, head, tail, mode):
        if mode=='head-batch':
            first_samples = head[:, 0, :].unsqueeze(1)
            
            dropout = nn.Dropout(p=self.args.ckdloss_dropout)
            dropped_samples = dropout(first_samples)
            
            part1 = head[:, :1, :]  # 第一个sample
            part2 = head[:, 1:, :]  # 第二个sample起的所有samples
            
            new_head = torch.cat([part1, dropped_samples, part2], dim=1)
            
            return new_head, tail
            
        elif mode=='tail-batch':
            first_samples = tail[:, 0, :].unsqueeze(1)
            
            dropout = nn.Dropout(p=self.args.ckdloss_dropout)
            dropped_samples = dropout(first_samples)
            
            part1 = tail[:, :1, :]  # 第一个sample
            part2 = tail[:, 1:, :]  # 第二个sample起的所有samples
            
            new_tail = torch.cat([part1, dropped_samples, part2], dim=1)
            
            return head, new_tail
        else:
            return None
                    
    def forward(self, data, subsampling_weight, mode):
        head, relation, tail, origin_relation = self.EmbeddingManager(data, mode)
        if self.args.cuda:
            head, relation, tail = head.cuda(), relation.cuda(), tail.cuda()
        t_score = self.KGE(head, origin_relation, tail, mode, self.args)
        
        head, tail = self.make_contrastive_pair(head, tail, mode)
        head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), self.EntityPruner(tail)
        if mode == 'head-batch':
            closs, closs_record = self.ContrastiveLoss(head, subsampling_weight)
        if mode == 'tail-batch':
            closs, closs_record = self.ContrastiveLoss(tail, subsampling_weight)
        
        score = self.KGE(head, relation, tail, mode, self.args)
        p_score, n_score = self.get_postive_score(score), self.get_negative_score(score)
        loss, loss_record = self.loss(p_score, n_score, subsampling_weight)
        KDloss, KDloss_record = self.KDLoss(t_score, self.delete_s_score(score), subsampling_weight)
        
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
        
        