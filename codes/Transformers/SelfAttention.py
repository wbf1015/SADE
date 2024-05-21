import sys
import os
import torch
import math
import torch.nn as nn
import numpy as np

transformerpath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(transformerpath)

from ScaleDotAttention import *
from PoswiseFeedForwardNet import *


'''
DimWiseAttention
早期使用的Transformer方法
'''
class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, V_dim, n_heads, residual='V'):
        super(SelfAttention, self).__init__()
        assert output_dim%n_heads==0
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.V_dim = V_dim
        
        self.n_heads = n_heads
        
        self.dk = output_dim//n_heads
        self.dv = self.dk
        
        self.residual = residual
        
        self.Wq = nn.Linear(input_dim, output_dim)
        self.Wk = nn.Linear(input_dim, output_dim)
        self.Wv = nn.Linear(V_dim, output_dim)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = nn.Linear(output_dim, output_dim)
        
    def forward(self, inputs_Q, inputs_K, inputs_V):
        
        batch_size, seq_len = inputs_Q.shape[0], inputs_Q.shape[1]
        
        q = self.Wq(inputs_Q).view(batch_size, seq_len, self.n_heads, self.dk, 1)
        k = self.Wk(inputs_K).view(batch_size, seq_len, self.n_heads, self.dk, 1)
        v = self.Wv(inputs_V)
        if self.residual == 'V':
            residual = v
        else:
            residual = inputs_V
            
        v = v.view(batch_size, seq_len, self.n_heads, self.dk, 1)
        
        out = self.ScaledDotProductAttention(q,k,v)
        out = out.view(batch_size, seq_len, self.output_dim)
        out = self.fc(out)
        
        if residual.shape == out.shape:
            return nn.LayerNorm(self.output_dim).cuda()(out + residual)
        else:
            return nn.LayerNorm(self.output_dim).cuda(out)
        
'''
把heads当成不同的tokens
SemAttention和StageAttention使用的Transformer方法
'''
class SelfAttention2(nn.Module):
    def __init__(self, input_dim, output_dim, V_dim, n_heads, residual='V'):
        super(SelfAttention2, self).__init__()
        assert output_dim%n_heads==0
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.V_dim = V_dim
        
        self.n_heads = n_heads
        
        self.dk = output_dim//n_heads
        self.dv = self.dk
        
        self.residual = residual
        
        self.Wq = nn.Linear(input_dim, output_dim)
        self.Wk = nn.Linear(input_dim, output_dim)
        self.Wv = nn.Linear(V_dim, output_dim)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = nn.Linear(output_dim, output_dim)
        
    def forward(self, inputs_Q, inputs_K, inputs_V):
        
        batch_size, seq_len = inputs_Q.shape[0], inputs_Q.shape[1]
        
        q = self.Wq(inputs_Q).view(batch_size, seq_len, self.n_heads, self.dk)
        k = self.Wk(inputs_K).view(batch_size, seq_len, self.n_heads, self.dk)
        v = self.Wv(inputs_V)
        if self.residual == 'V':
            residual = v
        else:
            residual = inputs_V
            
        v = v.view(batch_size, seq_len, self.n_heads, self.dk)
        
        out = self.ScaledDotProductAttention(q,k,v)
        out = out.view(batch_size, seq_len, self.output_dim)
        out = self.fc(out)
        
        if residual.shape == out.shape:
            return nn.LayerNorm(self.output_dim).cuda()(out + residual)
        else:
            return nn.LayerNorm(self.output_dim).cuda(out)


'''
把heads当成不同的tokens
并且把neg_sampling归到batch_size中
适配于NLPAttention
'''
class SelfAttention3(nn.Module):
    def __init__(self, input_dim, output_dim, V_dim, n_heads, residual='V'):
        super(SelfAttention3, self).__init__()
        assert output_dim%n_heads==0
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.V_dim = V_dim
        
        self.n_heads = n_heads
        
        self.dk = output_dim//n_heads
        self.dv = self.dk
        
        self.residual = residual
        
        self.Wq = nn.Linear(input_dim, output_dim)
        self.Wk = nn.Linear(input_dim, output_dim)
        self.Wv = nn.Linear(V_dim, output_dim)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = nn.Linear(output_dim, output_dim)
        
    def forward(self, inputs_Q, inputs_K, inputs_V):
        
        batch_size, seq_len = inputs_Q.shape[0], inputs_Q.shape[1]
        
        q = self.Wq(inputs_Q).view(batch_size*seq_len, self.n_heads, self.dk)
        k = self.Wk(inputs_K).view(batch_size*seq_len, self.n_heads, self.dk)
        v = self.Wv(inputs_V)
        if self.residual == 'V':
            residual = v
        else:
            residual = inputs_V
            
        v = v.view(batch_size*seq_len, self.n_heads, self.dk)
        
        out = self.ScaledDotProductAttention(q,k,v)
        out = out.view(batch_size, seq_len, self.output_dim)
        out = self.fc(out)
        
        if residual.shape == out.shape:
            return nn.LayerNorm(self.output_dim).cuda()(out + residual)
        else:
            return nn.LayerNorm(self.output_dim).cuda(out)


'''
TokenAttention设计的Self-Attention设计
输入和输出都是[batch-size, negsampling, seq_len, dim]
'''
class SelfAttention4(nn.Module):
    def __init__(self, input_dim, output_dim, V_dim, n_heads, residual='V'):
        super(SelfAttention4, self).__init__()
        assert output_dim%n_heads==0
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.V_dim = V_dim
        
        self.n_heads = n_heads
        self.dk = output_dim//n_heads
        self.dv = self.dk
        
        self.residual = residual
        
        self.Wq = nn.Linear(input_dim, output_dim)
        self.Wk = nn.Linear(input_dim, output_dim)
        self.Wv = nn.Linear(V_dim, output_dim)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = nn.Linear(output_dim, output_dim)
        
    def forward(self, inputs_Q, inputs_K, inputs_V): # [batch, neg_sampling, seq_len, dim]
        
        batch_size, neg_sampling, seq_len, dim = inputs_Q.shape[0], inputs_Q.shape[1], inputs_Q.shape[2], inputs_Q.shape[3]
        
        q = self.Wq(inputs_Q).view(batch_size*neg_sampling, self.n_heads, seq_len,self.dk)
        k = self.Wk(inputs_K).view(batch_size*neg_sampling, self.n_heads, seq_len,self.dk)
        v = self.Wv(inputs_V)
        if self.residual == 'V':
            residual = v
        else:
            residual = inputs_V
            
        v = v.view(batch_size*neg_sampling, self.n_heads, seq_len, self.dk)
        
        out = self.ScaledDotProductAttention(q,k,v)
        out = out.view(batch_size, neg_sampling, seq_len, self.output_dim)
        out = self.fc(out)
        
        if residual.shape == out.shape:
            return nn.LayerNorm(self.output_dim).cuda()(out + residual)
        else:
            return nn.LayerNorm(self.output_dim).cuda(out)


        
