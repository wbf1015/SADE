import sys
import os
import logging
import torch
import math
import torch.nn as nn
import numpy as np

CODEPATH = os.path.abspath(os.path.dirname(__file__))
CODEPATH = CODEPATH.rsplit('/', 1)[0]
sys.path.append(CODEPATH)

from Transformers.PoswiseFeedForwardNet import *
from Transformers.ScaleDotAttention import *
from Transformers.SelfAttention import *

class BasicStageAttention(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, d_ff=1):
        super(BasicStageAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.d_ff = d_ff                      
                
        self.local_extraction= SelfAttention2(input_dim//2, output_dim, input_dim, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init BasicStageAttention Pruner with input_dim={self.input_dim}, output_dim={self.output_dim}, n_heads={self.n_heads}, d_ff={self.d_ff}, local_extraction={self.local_extraction.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')
    
        
    def forward(self, enc_inputs, forget):
        local_inputs1 = enc_inputs[:, :, :self.input_dim//2]
        local_inputs2 = enc_inputs[:, :, self.input_dim//2:]
        outputs = self.local_extraction(local_inputs1, local_inputs2, forget)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_) # 加还是*都可以试试 StageAttention1/2
        # # forget_ = outputs + forget_
        # forget_ = outputs + outputs * self.sigmoid(forget_)
        
        return outputs, forget_


class StageAttention(nn.Module):
    def __init__(self, args):
        super(StageAttention, self).__init__()
        self.args = args
        self.layers = nn.ModuleList([
            BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff),
            BasicStageAttention(256, 64, args.head2, d_ff=args.t_dff),
        ])
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        for layer in self.layers:
            outputs, forget = layer(outputs, forget)
        return outputs 


class StageAttention2(nn.Module):
    def __init__(self, args):
        super(StageAttention2, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicStageAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        tmp_embedding = None
        
        outputs, forget = self.layer1(outputs, forget)
        tmp_embedding = outputs.clone()
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs, tmp_embedding


class RelationStageAttention(nn.Module):
    def __init__(self, args):
        super(RelationStageAttention, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(256, 128, args.head1//2, d_ff=args.t_dff)
        self.layer2 = BasicStageAttention(128, 32, args.head2//2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        tmp_embedding = None
        
        outputs, forget = self.layer1(outputs, forget)
        tmp_embedding = outputs.clone()
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs, tmp_embedding


if __name__ == '__main__':
    Attention =  StageAttention(1024, 512, 8, 4)