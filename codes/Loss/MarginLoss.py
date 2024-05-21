import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class MarginLoss(nn.Module):

	def __init__(self, adv_temperature = None, margin = 6.0):
		super(MarginLoss, self).__init__()
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score, subsampling_weight=None):
		if self.adv_flag:
			loss = (self.get_weights(n_score) * torch.max(p_score.unsqueeze(1) - n_score, -self.margin)).sum(-1)
			if subsampling_weight!=None:
				loss = (loss*subsampling_weight).mean() + self.margin
			else:
				loss = (loss).mean() + self.margin
		else:
			loss = torch.max(p_score - n_score, -self.margin)
			if subsampling_weight!=None:
				loss = (loss*subsampling_weight).mean() + self.margin
			else:
				loss = (loss).mean() + self.margin

		loss_record = {
            'hard_loss': loss.item(),
        }
		return loss, loss_record
			
	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()