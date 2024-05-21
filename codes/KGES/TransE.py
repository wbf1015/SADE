import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class TransE(nn.Module):
    def __init__(self, embedding_range=None, embedding_dim=None, margin=None):
        super(TransE, self).__init__()
        self.margin = margin
        
        logging.info(f'Init TransE with margin={self.margin}')

    def forward(self, head, relation, tail, mode, args=None):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = torch.norm(score, p=1, dim=2)
        
        if self.margin is not None:
            score = self.margin - score
        else:
            score = score
        
        return score
    
    def get_distance(self, head, relation, tail, mode, args):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        return score


    def get_index(self, dim_score, args):
        assert (len(dim_score)%2)==0
        # 在前512维中找到分数最小的32个维度的下标
        indices = torch.topk(dim_score, args.target_dim, largest=False).indices # 从小到大排序

        return indices