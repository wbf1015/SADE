import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class ComplEx(nn.Module):
    def __init__(self, embedding_range=None, embedding_dim=None, margin=None):
        super(ComplEx, self).__init__()
        self.margin = margin
        
        logging.info(f'Init ComplEx with margin={self.margin}')

    def forward(self, head, relation, tail, mode, real_para=None):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        
        if self.margin != None:
            score = self.margin + score
        
        return score
