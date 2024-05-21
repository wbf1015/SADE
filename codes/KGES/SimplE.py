import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class SimplE(nn.Module):
    def __init__(self, margin=None):
        super(SimplE, self).__init__()
        self.margin = margin
        
        logging.info(f'Init SimplE with margin={self.margin}')

    def forward(self, head, relation, tail, mode, real_para=None):
        head_head, head_tail = torch.chunk(head, 2, dim=2)
        tail_head, tail_tail = torch.chunk(tail, 2, dim=2)
        r_embs, r_inv_embs = torch.chunk(relation, 2, dim=2)
        
        scores1 = torch.sum(head_head * r_embs * tail_tail, dim=-1)
        scores2 = torch.sum(tail_head * r_inv_embs * head_tail, dim=-1)
        
        if self.margin != None:
            score = self.margin - ((scores1 + scores2) / 2)
        else:
            score = (scores1 + scores2) / 2
        
        return score