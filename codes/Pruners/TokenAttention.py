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

class BasicTokenAttention(nn.Module):
    def __init__(self, input_dim, output_dim, n_token, n_heads, d_ff=1, LN=False):
        super(BasicTokenAttention, self).__init__()
        assert (input_dim%n_token==0) and (output_dim%n_token==0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_token = n_token
        self.n_heads = n_heads
        self.d_ff = d_ff       
        self.LN = LN
        
        self.sem_transform=SelfAttention4(input_dim//n_token, output_dim//n_token, input_dim//n_token, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init BasicTokenAttention Pruner with input_dim={self.input_dim}, output_dim={self.output_dim}, n_heads={self.n_heads}, d_ff={self.d_ff}, sem_transform={self.sem_transform.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')

    def forward(self, enc_inputs, forget):
        batch_size, neg_sampling = enc_inputs.shape[0], enc_inputs.shape[1]
        inputs_Q, inputs_K, inputs_V = enc_inputs.view(batch_size, neg_sampling, self.n_token, self.input_dim//self.n_token), enc_inputs.view(batch_size, neg_sampling, self.n_token, self.input_dim//self.n_token), forget.view(batch_size, neg_sampling, self.n_token, self.input_dim//self.n_token)
        outputs = self.sem_transform(inputs_Q, inputs_K, inputs_V)
        outputs = outputs.view(batch_size, neg_sampling, self.output_dim)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_)
        
        return outputs, forget_
        
class BasicTokenAttention2(nn.Module):
    def __init__(self, input_dim, output_dim, n_token, n_heads, d_ff=1, LN=False):
        super(BasicTokenAttention2, self).__init__()
        assert (input_dim%n_token==0) and (output_dim%n_token==0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_token = n_token
        self.n_heads = n_heads
        self.d_ff = d_ff       
        self.LN = LN
        
        self.sem_transform=SelfAttention4(input_dim//n_token, output_dim, input_dim//n_token, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init BasicTokenAttention2 Pruner with input_dim={self.input_dim}, output_dim={self.output_dim}, n_heads={self.n_heads}, d_ff={self.d_ff}, sem_transform={self.sem_transform.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')

    def forward(self, enc_inputs, forget):
        batch_size, neg_sampling = enc_inputs.shape[0], enc_inputs.shape[1]
        inputs_Q, inputs_K, inputs_V = enc_inputs.view(batch_size, neg_sampling, self.n_token, self.input_dim//self.n_token), enc_inputs.view(batch_size, neg_sampling, self.n_token, self.input_dim//self.n_token), forget.view(batch_size, neg_sampling, self.n_token, self.input_dim//self.n_token)
        outputs = self.sem_transform(inputs_Q, inputs_K, inputs_V)
        outputs = outputs.mean(dim=-2)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_)
        
        return outputs, forget_


class BasicSemAttention(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, d_ff=1, LN=False):
        super(BasicSemAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.d_ff = d_ff       
        self.LN = LN                
        
        self.sem_transform= SelfAttention2(input_dim, output_dim, input_dim, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init BasicSemAttention Pruner with input_dim={self.input_dim}, output_dim={self.output_dim}, n_heads={self.n_heads}, d_ff={self.d_ff}, sem_transform={self.sem_transform.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')
    
    def forward(self, enc_inputs, forget):
        outputs = self.sem_transform(enc_inputs, enc_inputs, forget)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_)
        
        return outputs, forget_


class TokenAttention(nn.Module):
    def __init__(self, args):
        super(TokenAttention, self).__init__()
        self.args = args
        self.layer1 = BasicTokenAttention(1024, 256, args.token1, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs


class TokenAttention2(nn.Module):
    def __init__(self, args):
        super(TokenAttention2, self).__init__()
        self.args = args
        self.layer1 = BasicTokenAttention2(1024, 256, args.token1, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs