import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class RotatE(nn.Module):
    def __init__(self, margin=None, embedding_range=11.0, embedding_dim=512):
        super(RotatE, self).__init__()
        self.margin = margin
        self.embedding_range = embedding_range
        self.embedding_dim = embedding_dim
        
        logging.info(f'Init RotatE with embedding_range={self.embedding_range}, embedding_dim={self.embedding_dim}, margin={self.margin}')
    
    def forward(self, head, relation, tail, mode, args):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        
        if head.shape[-1]>500:
            phase_relation = relation/(((self.embedding_range)/self.embedding_dim)/pi)
        else:
            embedding_range, embedding_dim = 2.0+args.gamma, args.target_dim
            phase_relation = relation/(((embedding_range)/embedding_dim)/pi)
        
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = score.sum(dim = 2)
        
        if self.margin is not None:
            score = self.margin - score
        else:
            score = score
        
        return score 
    
    
    def get_distance(self, head, relation, tail, mode, args):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=0)
        re_tail, im_tail = torch.chunk(tail, 2, dim=0)

        #Make phases of relations uniformly distributed in [-pi, pi]
        
        if head.shape[-1]>500:
            phase_relation = relation/(((self.embedding_range)/self.embedding_dim)/pi)
        else:
            embedding_range, embedding_dim = 2.0+args.gamma, args.target_dim
            phase_relation = relation/(((embedding_range)/embedding_dim)/pi)
        
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        
        distance = torch.cat((re_score, im_score), dim=0)
        return distance


    def get_index(self, dim_score, args):
        assert (len(dim_score)%2)==0
        # 前512维
        first_half, second_half = dim_score[:len(dim_score)//2], dim_score[len(dim_score)//2:]

        # 在前512维中找到分数最小的32个维度的下标
        indices_first_half = torch.topk(first_half, args.target_dim, largest=False).indices # 从小到大排序

        # 在后512维中找到分数最小的32个维度的下标
        indices_second_half = torch.topk(second_half, args.target_dim, largest=False).indices

        # 调整后512维的下标，使其相对于整个1024维向量
        indices_second_half += 512

        # 合并两组下标
        final_indices = torch.cat((indices_first_half, indices_second_half))

        return final_indices