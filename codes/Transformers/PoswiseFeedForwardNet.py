import sys
import torch
import math
import torch.nn as nn
import numpy as np

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, input_dim, d_ff=1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.input_dim = input_dim
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(input_dim, d_ff*input_dim, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff*input_dim, input_dim, bias=False))
        
    def forward(self, inputs):                             # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.input_dim).cuda()(output + residual)   # [batch_size, seq_len, d_model]  


class PoswiseFeedForwardNet2(nn.Module):
    def __init__(self, input_dim, d_ff=1):
        super(PoswiseFeedForwardNet2, self).__init__()
        self.input_dim = input_dim
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(input_dim, d_ff*input_dim, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff*input_dim, input_dim, bias=False))
        
    def forward(self, inputs):                             # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return output+residual