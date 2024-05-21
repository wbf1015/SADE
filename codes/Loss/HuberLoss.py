import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class HuberLoss(nn.Module):
    def __init__(self, adv_temperature = None, margin = 6.0, KGEmargin = 6.0, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.margin = nn.Parameter(torch.Tensor([margin])) # 在这里用不到
        self.KGEmargin = KGEmargin
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
            logging.info(f'Init KDLoss with adv_temperature={adv_temperature}')
        else:
            self.adv_flag = False
            logging.info('Init KDLoss without adv_temperature')

    '''
    reference:https://juejin.cn/s/pytorch%20huber%20loss%20example
    '''
    def huber_loss(self, t_score, s_score, subsampling_weight=None):
        residual = torch.abs(t_score - s_score)
        condition = (residual < self.delta).float()
        loss = condition * 0.5 * residual**2 + (1 - condition) * (self.delta * residual - 0.5 * self.delta**2)
               
        p_loss, n_loss = loss[:, 0], loss[:, 1:]
        
        if self.adv_flag:
            n_loss = (F.softmax(n_loss * self.adv_temperature, dim = 1).detach() 
                              * n_loss).sum(dim = 1)
        else:
            n_loss = n_loss.mean(dim = 1)
        p_loss = p_loss
        
        if subsampling_weight!=None:
            p_loss = p_loss.mean()
            n_loss = n_loss.mean()
        else:
            p_loss = (subsampling_weight * p_loss).sum()/subsampling_weight.sum()
            n_loss = (subsampling_weight * n_loss).sum()/subsampling_weight.sum()
        
        loss = (p_loss + n_loss)/2
        
        return loss, p_loss, n_loss
        
    
    def forward(self, t_score, s_score, subsampling_weight=None):
        if self.KGEmargin is not None:
            t_score, s_score = self.KGEmargin-t_score, self.KGEmargin-s_score
        
        loss, p_loss, n_loss = self.huber_loss(t_score, s_score, subsampling_weight=subsampling_weight)
        
        loss_record = {
            'positive_huber_loss': p_loss.item(),
            'negative_huber_loss': n_loss.item(),
            'huber_loss': loss.item(),
        }
        
        return loss, loss_record


    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()