import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class SigmoidLossOrigin(nn.Module):
    def __init__(self, adv_temperature = None, margin = 6.0):
        super(SigmoidLossOrigin, self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
            logging.info(f'Init SigmoidLossOrigin with adv_temperature={adv_temperature}')
        else:
            self.adv_flag = False
            logging.info('Init SigmoidLossOrigin without adv_temperature')

    def forward(self, p_score, n_score, subsampling_weight=None, sub_margin=False, add_margin=False):
        if sub_margin:
            p_score, n_score = self.margin-p_score, self.margin-n_score
        if add_margin:
            p_score, n_score = self.margin+p_score, self.margin+n_score
        if self.adv_flag:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(n_score * self.adv_temperature, dim = 1).detach() 
                              * F.logsigmoid(-n_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-n_score).mean(dim = 1)
        
        positive_score = F.logsigmoid(p_score)

        if subsampling_weight!=None:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
                
        # 到这里就是1*1的了
        loss = (positive_sample_loss + negative_sample_loss)/2
        
        loss_record = {
            'hard_positive_sample_loss': positive_sample_loss.item(),
            'hard_negative_sample_loss': negative_sample_loss.item(),
            'hard_loss': loss.item(),
        }
        
        return loss, loss_record

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()