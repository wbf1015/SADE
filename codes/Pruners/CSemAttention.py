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

def make_contrastive_pair(inputs, dropout):
    first_samples = inputs[:, 0, :].unsqueeze(1)
    
    dropout = nn.Dropout(p=dropout)
    dropped_samples = dropout(first_samples)
    
    part1 = inputs[:, :1, :]  # 第一个sample
    part2 = inputs[:, 1:, :]  # 第二个sample起的所有samples
    
    new_inputs = torch.cat([part1, dropped_samples, part2], dim=1)
    
    return new_inputs

def del_pos_ent(inputs):
    return torch.cat((inputs[:, :1], inputs[:, 2:]), dim=1)

class LGSemAttention(nn.Module):
    def __init__(self, input_dim, output_dim, subspace, n_heads, d_ff=1, LN=False, dropout=0.0):
        super(LGSemAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.subspace = subspace
        self.n_heads = n_heads
        self.d_ff = d_ff                      
        self.LN = LN
        self.Dropout = nn.Dropout(p=dropout)
        
        self.subspace_sem = nn.ModuleList([
            nn.Linear(input_dim // subspace, output_dim)
            for _ in range(subspace)
        ])
        self.global_sem = nn.Linear(subspace*output_dim+input_dim, output_dim)
        
        self.sem_fusion= SelfAttention4(output_dim, output_dim, output_dim, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init LGSemAttention with input_dim={self.input_dim}, output_dim={self.output_dim}, subspace={self.subspace}, n_heads={self.n_heads}, d_ff={self.d_ff}, dropout={dropout},sem_fusion={self.sem_fusion.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')

    def get_sem(self, enc_inputs):
        batch_size, neg_sampling, dim = enc_inputs.shape
        
        # 分割enc_inputs
        chunks = torch.chunk(enc_inputs, self.subspace, dim=-1)
        
        # 通过subspace_sem处理每一份
        processed_chunks = [layer(chunk).unsqueeze(-2) for chunk, layer in zip(chunks, self.subspace_sem)]

        # 将结果合并为所需形状
        combined = torch.cat(processed_chunks, dim=-2)
        
        # 保存副本并重塑
        reshaped_combined = combined.view(batch_size, neg_sampling, -1)
        
        # 与原始输入拼接并通过global_sem处理
        global_input = torch.cat((reshaped_combined, enc_inputs), dim=-1)
        global_output = self.global_sem(global_input)
        
        
        Semantic = torch.cat((combined, global_output.unsqueeze(2)), dim=2)
        
        return Semantic
    
    
    def forward(self, enc_inputs, forget):
        semantic = self.get_sem(enc_inputs)
        outputs = self.sem_fusion(self.Dropout(semantic), self.Dropout(semantic), self.Dropout(semantic))
        outputs = outputs.mean(dim=-2)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_) # 加还是*都可以试试 StageAttention1/2
        # # forget_ = outputs + forget_
        # forget_ = outputs + outputs * self.sigmoid(forget_)
        
        return outputs, forget_


class SemUpdate(nn.Module):
    def __init__(self, sem_dim, embedding_dim, LN=False, dropout=0.0):
        super(SemUpdate, self).__init__()
        self.sem_dim = sem_dim
        self.embedding_dim = embedding_dim
        self.LN = LN
        self.Dropout = nn.Dropout(p=dropout)
        
        self.reset_weight = nn.Linear(sem_dim+embedding_dim, sem_dim)
        self.update_weight = nn.Linear(sem_dim+embedding_dim, sem_dim)
        self.reset_transfer = nn.Linear(sem_dim, sem_dim)
        self.update = nn.Linear(sem_dim+embedding_dim, sem_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        logging.info(f'Init SemUpdate with sem_dim={self.sem_dim}, embedding_dim={self.embedding_dim}, LN={self.LN}, dropout={dropout}')
        
    def forward(self, sem, origin_embedding):
        reset = self.sigmoid(self.reset_weight(torch.cat((self.Dropout(sem), origin_embedding), dim=-1)))
        update = self.sigmoid(self.update_weight(torch.cat((self.Dropout(sem), origin_embedding), dim=-1)))
        
        h = self.tanh(self.update(torch.cat((origin_embedding, self.reset_transfer(self.Dropout(sem))*reset), dim=-1)))
        outputs = (1-update) * sem + update * h

        if self.LN:
            return nn.LayerNorm(self.sem_dim).cuda()(outputs) 
        else:
            return outputs


class LowDimGenerate(nn.Module):
    def __init__(self, sem_dim, embedding_dim, target_dim, d_ff=1, LN=False, dropout=0.0):
        super(LowDimGenerate, self).__init__()
        self.sem_dim = sem_dim
        self.embedding_dim = embedding_dim
        self.target_dim = target_dim
        self.d_ff = d_ff
        self.LN = LN
        self.Dropout = nn.Dropout(p=dropout)
        
        self.Basic_Position = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//4),
            nn.Linear(embedding_dim//4, target_dim)
        )
        
        self.FT1 = nn.Linear(sem_dim+target_dim, target_dim//2)
        self.FT2 = nn.Linear(sem_dim+target_dim, target_dim//2)
        self.FTALL = nn.Linear(sem_dim+target_dim, target_dim)
        
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(target_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(target_dim, d_ff=d_ff)
        
        logging.info(f'Init LowDimGenerate with sem_dim={self.sem_dim}, embedding_dim={self.embedding_dim}, target_dim={self.target_dim}, self.d_ff={self.d_ff}, LN={self.LN}, dropout={dropout}')
    
    def forward(self, sem, origin_embedding):
        basic_position = self.Basic_Position(origin_embedding)
        FT1 = self.FT1(torch.cat((self.Dropout(sem), basic_position), dim=-1))
        FT2 = self.FT2(torch.cat((self.Dropout(sem), basic_position), dim=-1))
        ft_position = basic_position + torch.cat((FT1, FT2), dim=-1)
        
        FTALL = self.FTALL(torch.cat((self.Dropout(sem), ft_position), dim=-1))
        ft_position = ft_position + FTALL
        
        outputs = self.fc(ft_position)
        
        return outputs


class CSemAttention8(nn.Module):
    def __init__(self, args, ContrastiveLoss):
        super(CSemAttention8, self).__init__()
        self.args = args
        self.ContrastiveLoss = ContrastiveLoss
        self.layer1 = LGSemAttention(1024, 256, args.token1, args.head1, d_ff=args.t_dff, dropout=args.dropout)
        self.layer2 = SemUpdate(256, 1024, dropout=args.dropout)
        self.layer3 = LowDimGenerate(256, 1024, 60, args.t_dff, dropout=args.dropout)
    
    
    def cal_loss_del(self, inputs, subsampling_weight):
        closs, closs_record = self.ContrastiveLoss(inputs, subsampling_weight)
        new_inputs = del_pos_ent(inputs)
        
        return new_inputs, closs, closs_record
        
    
    def forward(self, inputs, flag=False, subsampling_weight=None):
        if flag is True:
            new_inputs = make_contrastive_pair(inputs, self.args.ckdloss_dropout)
            outputs, _ = self.layer1(new_inputs, new_inputs)
            outputs = self.layer2(outputs, new_inputs)
            new_outputs, closs, closs_record = self.cal_loss_del(outputs, subsampling_weight)
            outputs = self.layer3(new_outputs, inputs)
            
            return outputs, closs, closs_record
            
        if flag is False:
            outputs, _ = self.layer1(inputs, inputs)
            outputs = self.layer2(outputs, inputs)
            outputs = self.layer3(outputs, inputs)
        
            return outputs


class CSemAttention8TransE(nn.Module):
    def __init__(self, args, ContrastiveLoss):
        super(CSemAttention8TransE, self).__init__()
        self.args = args
        self.ContrastiveLoss = ContrastiveLoss
        self.layer1 = LGSemAttention(1024, 256, args.token1, args.head1, d_ff=args.t_dff, dropout=args.dropout)
        self.layer2 = SemUpdate(256, 1024, dropout=args.dropout)
        self.layer3 = LowDimGenerate(256, 1024, 32, args.t_dff, dropout=args.dropout)
    
    
    def cal_loss_del(self, inputs, subsampling_weight):
        closs, closs_record = self.ContrastiveLoss(inputs, subsampling_weight)
        new_inputs = del_pos_ent(inputs)
        
        return new_inputs, closs, closs_record
        
    
    def forward(self, inputs, flag=False, subsampling_weight=None):
        if flag is True:
            new_inputs = make_contrastive_pair(inputs, self.args.ckdloss_dropout)
            outputs, _ = self.layer1(new_inputs, new_inputs)
            outputs = self.layer2(outputs, new_inputs)
            new_outputs, closs, closs_record = self.cal_loss_del(outputs, subsampling_weight)
            outputs = self.layer3(new_outputs, inputs)
            
            return outputs, closs, closs_record
            
        if flag is False:
            outputs, _ = self.layer1(inputs, inputs)
            outputs = self.layer2(outputs, inputs)
            outputs = self.layer3(outputs, inputs)
        
            return outputs


class CSemAttention8TransE2(nn.Module):
    def __init__(self, args, ContrastiveLoss):
        super(CSemAttention8TransE2, self).__init__()
        self.args = args
        self.ContrastiveLoss = ContrastiveLoss
        self.layer1 = LGSemAttention(1024, 256, args.token1, args.head1, d_ff=args.t_dff, dropout=args.dropout)
        self.layer2 = SemUpdate(256, 1024, dropout=args.dropout)
        self.layer3 = LowDimGenerate(256, 1024, 32, args.t_dff, dropout=args.dropout)
    
    
    def cal_loss_del(self, inputs, subsampling_weight):
        closs, closs_record = self.ContrastiveLoss(inputs, subsampling_weight)
        new_inputs = del_pos_ent(inputs)
        
        return new_inputs, closs, closs_record
        
    
    def forward(self, inputs, flag=False, subsampling_weight=None):
        if flag is True:
            new_inputs = make_contrastive_pair(inputs, self.args.ckdloss_dropout)
            outputs, _ = self.layer1(new_inputs, new_inputs)
            outputs = self.layer2(outputs, new_inputs)
            new_outputs, closs, closs_record = self.cal_loss_del(outputs, subsampling_weight)
            outputs = self.layer3(new_outputs, inputs)
            
            return outputs, closs, closs_record
            
        if flag is False:
            outputs, _ = self.layer1(inputs, inputs)
            outputs = self.layer2(outputs, inputs)
            outputs = self.layer3(outputs, inputs)
        
            return outputs


class CSemAttention8_2(nn.Module):
    def __init__(self, args, ContrastiveLoss):
        super(CSemAttention8_2, self).__init__()
        self.args = args
        self.ContrastiveLoss = ContrastiveLoss
        self.layer1 = LGSemAttention(1024, 128, args.token1, args.head1, d_ff=args.t_dff, dropout=args.dropout)
        self.layer2 = SemUpdate(128, 1024, dropout=args.dropout)
        self.layer3 = LowDimGenerate(128, 1024, 64, args.t_dff, dropout=args.dropout)
    
    
    def cal_loss_del(self, inputs, subsampling_weight):
        closs, closs_record = self.ContrastiveLoss(inputs, subsampling_weight)
        new_inputs = del_pos_ent(inputs)
        
        return new_inputs, closs, closs_record
        
    
    def forward(self, inputs, flag=False, subsampling_weight=None):
        if flag is True:
            new_inputs = make_contrastive_pair(inputs, self.args.ckdloss_dropout)
            outputs, _ = self.layer1(new_inputs, new_inputs)
            outputs = self.layer2(outputs, new_inputs)
            new_outputs, closs, closs_record = self.cal_loss_del(outputs, subsampling_weight)
            outputs = self.layer3(new_outputs, inputs)
            
            return outputs, closs, closs_record
            
        if flag is False:
            outputs, _ = self.layer1(inputs, inputs)
            outputs = self.layer2(outputs, inputs)
            outputs = self.layer3(outputs, inputs)
        
            return outputs


class CSemAttention8_3(nn.Module):
    def __init__(self, args, ContrastiveLoss):
        super(CSemAttention8_3, self).__init__()
        self.args = args
        self.ContrastiveLoss = ContrastiveLoss
        self.layer1 = LGSemAttention(1024, 64, args.token1, args.head1, d_ff=args.t_dff, dropout=args.dropout)
        self.layer2 = SemUpdate(64, 1024, dropout=args.dropout)
        self.layer3 = LowDimGenerate(64, 1024, 64, args.t_dff, dropout=args.dropout)
    
    
    def cal_loss_del(self, inputs, subsampling_weight):
        closs, closs_record = self.ContrastiveLoss(inputs, subsampling_weight)
        new_inputs = del_pos_ent(inputs)
        
        return new_inputs, closs, closs_record
        
    
    def forward(self, inputs, flag=False, subsampling_weight=None):
        if flag is True:
            new_inputs = make_contrastive_pair(inputs, self.args.ckdloss_dropout)
            outputs, _ = self.layer1(new_inputs, new_inputs)
            outputs = self.layer2(outputs, new_inputs)
            new_outputs, closs, closs_record = self.cal_loss_del(outputs, subsampling_weight)
            outputs = self.layer3(new_outputs, inputs)
            
            return outputs, closs, closs_record
            
        if flag is False:
            outputs, _ = self.layer1(inputs, inputs)
            outputs = self.layer2(outputs, inputs)
            outputs = self.layer3(outputs, inputs)
        
            return outputs


class CSemAttention9(nn.Module):
    def __init__(self, args, ContrastiveLoss):
        super(CSemAttention9, self).__init__()
        self.args = args
        self.ContrastiveLoss = ContrastiveLoss
        self.layer1 = LGSemAttention(1024, 256, args.token1, args.head1, d_ff=args.t_dff, dropout=args.dropout)
        self.layer2 = SemUpdate(256, 1024, dropout=args.dropout)
        self.layer3 = LowDimGenerate(256, 1024, 64, args.t_dff, dropout=args.dropout)
    
    
    def cal_loss_del(self, inputs, subsampling_weight):
        closs, closs_record = self.ContrastiveLoss(inputs, subsampling_weight)
        new_inputs = del_pos_ent(inputs)
        
        return new_inputs, closs, closs_record
        
    
    def forward(self, inputs, flag=False, subsampling_weight=None):
        if flag is True:
            new_inputs = make_contrastive_pair(inputs, self.args.ckdloss_dropout)
            outputs, _ = self.layer1(new_inputs, new_inputs)
            new_outputs, closs, closs_record = self.cal_loss_del(outputs, subsampling_weight)
            outputs = self.layer2(new_outputs, inputs)
            outputs = self.layer3(outputs, inputs)
            
            return outputs, closs, closs_record
            
        if flag is False:
            outputs, _ = self.layer1(inputs, inputs)
            outputs = self.layer2(outputs, inputs)
            outputs = self.layer3(outputs, inputs)
        
            return outputs


class Ablation_SEM(nn.Module):
    def __init__(self, args, ContrastiveLoss):
        super(Ablation_SEM, self).__init__()
        self.args = args
        self.ContrastiveLoss = ContrastiveLoss
        self.MLP1 = nn.Linear(512, 128)
        self.MLP2 = nn.Linear(128, 32)
    
    
    def cal_loss_del(self, inputs, subsampling_weight):
        closs, closs_record = self.ContrastiveLoss(inputs, subsampling_weight)
        new_inputs = del_pos_ent(inputs)
        
        return new_inputs, closs, closs_record
        
    
    def forward(self, inputs, flag=False, subsampling_weight=None):
        if flag is True:
            new_inputs = make_contrastive_pair(inputs, self.args.ckdloss_dropout)
            outputs = self.MLP1(new_inputs)
            new_outputs, closs, closs_record = self.cal_loss_del(outputs, subsampling_weight)
            outputs = self.MLP2(new_outputs)
            
            return outputs, closs, closs_record
            
        if flag is False:
            outputs = self.MLP1(inputs)
            outputs = self.MLP2(outputs)
        
            return outputs



class Ablation_AUG(nn.Module):
    def __init__(self, args, ContrastiveLoss):
        super(Ablation_AUG, self).__init__()
        self.args = args
        self.ContrastiveLoss = ContrastiveLoss
        self.layer1 = LGSemAttention(512, 128, args.token1, args.head1, d_ff=args.t_dff, dropout=args.dropout)
        self.layer3 = LowDimGenerate(128, 512, 32, args.t_dff, dropout=args.dropout)
    
    
    def cal_loss_del(self, inputs, subsampling_weight):
        closs, closs_record = self.ContrastiveLoss(inputs, subsampling_weight)
        new_inputs = del_pos_ent(inputs)
        
        return new_inputs, closs, closs_record
        
    
    def forward(self, inputs, flag=False, subsampling_weight=None):
        if flag is True:
            new_inputs = make_contrastive_pair(inputs, self.args.ckdloss_dropout)
            outputs, _ = self.layer1(new_inputs, new_inputs)
            new_outputs, closs, closs_record = self.cal_loss_del(outputs, subsampling_weight)
            outputs = self.layer3(new_outputs, inputs)
            
            return outputs, closs, closs_record
            
        if flag is False:
            outputs, _ = self.layer1(inputs, inputs)
            outputs = self.layer3(outputs, inputs)
        
            return outputs