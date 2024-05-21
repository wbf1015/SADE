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
    def __init__(self, input_dim, output_dim, n_heads, d_ff=1, LN=False):
        super(BasicStageAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.d_ff = d_ff                      
        self.LN = LN        
                
        self.local_extraction= SelfAttention2(input_dim//2, output_dim, input_dim, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
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


class BasicStageAttention2(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, d_ff=1, LN=False):
        super(BasicStageAttention2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.d_ff = d_ff                      
        self.LN = LN        
        
        self.modality1 = nn.Linear(input_dim, input_dim//2)
        self.modality2 = nn.Linear(input_dim, input_dim//2)
        self.local_extraction= SelfAttention2(input_dim//2, output_dim, input_dim, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init BasicStageAttention Pruner with input_dim={self.input_dim}, output_dim={self.output_dim}, n_heads={self.n_heads}, d_ff={self.d_ff}, local_extraction={self.local_extraction.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')
    
        
    def forward(self, enc_inputs, forget):
        local_inputs1 = self.modality1(enc_inputs)
        local_inputs2 = self.modality2(enc_inputs)
        outputs = self.local_extraction(local_inputs1, local_inputs2, forget)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_) # 加还是*都可以试试 StageAttention1/2
        # # forget_ = outputs + forget_
        # forget_ = outputs + outputs * self.sigmoid(forget_)
        
        return outputs, forget_


class NLPStageAttention(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, d_ff=1, LN=False):
        super(NLPStageAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.d_ff = d_ff                      
        self.LN = LN        
                
        self.local_extraction= SelfAttention3(input_dim, output_dim, input_dim, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init NLPAttention Pruner with input_dim={self.input_dim}, output_dim={self.output_dim}, n_heads={self.n_heads}, d_ff={self.d_ff}, local_extraction={self.local_extraction.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')
    
        
    def forward(self, enc_inputs, forget):
        outputs = self.local_extraction(enc_inputs, enc_inputs, forget)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_) # 加还是*都可以试试 StageAttention1/2
        # # forget_ = outputs + forget_
        # forget_ = outputs + outputs * self.sigmoid(forget_)
        
        return outputs, forget_


class LGSemAttention(nn.Module):
    def __init__(self, input_dim, output_dim, subspace, n_heads, d_ff=1, LN=False):
        super(LGSemAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.subspace = subspace
        self.n_heads = n_heads
        self.d_ff = d_ff                      
        self.LN = LN        
        
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
        
        logging.info(f'Init LGSemAttention with input_dim={self.input_dim}, output_dim={self.output_dim}, subspace={self.subspace}, n_heads={self.n_heads}, d_ff={self.d_ff}, sem_fusion={self.sem_fusion.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')

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
        outputs = self.sem_fusion(semantic, semantic, semantic)
        outputs = outputs.mean(dim=-2)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_) # 加还是*都可以试试 StageAttention1/2
        # # forget_ = outputs + forget_
        # forget_ = outputs + outputs * self.sigmoid(forget_)
        
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


class SemUpdate(nn.Module):
    def __init__(self, sem_dim, embedding_dim, LN=False):
        super(SemUpdate, self).__init__()
        self.sem_dim = sem_dim
        self.embedding_dim = embedding_dim
        self.LN = LN        
        
        self.reset_weight = nn.Linear(sem_dim+embedding_dim, sem_dim)
        self.update_weight = nn.Linear(sem_dim+embedding_dim, sem_dim)
        self.reset_transfer = nn.Linear(sem_dim, sem_dim)
        self.update = nn.Linear(sem_dim+embedding_dim, sem_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        logging.info(f'Init SemUpdate with sem_dim={self.sem_dim}, embedding_dim={self.embedding_dim}, LN={self.LN}')
        
    def forward(self, sem, origin_embedding):
        reset = self.sigmoid(self.reset_weight(torch.cat((sem, origin_embedding), dim=-1)))
        update = self.sigmoid(self.update_weight(torch.cat((sem, origin_embedding), dim=-1)))
        
        h = self.tanh(self.update(torch.cat((origin_embedding, self.reset_transfer(sem)*reset), dim=-1)))
        outputs = (1-update) * sem + update * h

        if self.LN:
            return nn.LayerNorm(self.sem_dim).cuda()(outputs) 
        else:
            return outputs


class LowDimGenerate(nn.Module):
    def __init__(self, sem_dim, embedding_dim, target_dim, d_ff=1, LN=False):
        super(LowDimGenerate, self).__init__()
        self.sem_dim = sem_dim
        self.embedding_dim = embedding_dim
        self.target_dim = target_dim
        self.d_ff = d_ff
        self.LN = LN
        
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
    
    def forward(self, sem, origin_embedding):
        basic_position = self.Basic_Position(origin_embedding)
        FT1 = self.FT1(torch.cat((sem, basic_position), dim=-1))
        FT2 = self.FT2(torch.cat((sem, basic_position), dim=-1))
        ft_position = basic_position + torch.cat((FT1, FT2), dim=-1)
        
        FTALL = self.FTALL(torch.cat((sem, ft_position), dim=-1))
        ft_position = ft_position + FTALL
        
        outputs = self.fc(ft_position)
        
        return outputs
        

'''
前一半后一半用来从几何特征提取语义特征
然后都使用self-attention
'''
class SemAttention(nn.Module):
    def __init__(self, args):
        super(SemAttention, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs


'''
在中间的语义变换期间加上LayerNorm
'''
class SemAttentionLN(nn.Module):
    def __init__(self, args):
        super(SemAttentionLN, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs


class SemAttentionLNTransE(nn.Module):
    def __init__(self, args):
        super(SemAttentionLNTransE, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(512, 128, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(128, 32, args.head2, d_ff=args.t_dff, LN=True)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs


'''
语义信息的维度递减
'''
class SemAttention2(nn.Module):
    def __init__(self, args):
        super(SemAttention2, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 128, args.head2, d_ff=args.t_dff)
        self.layer3 = BasicSemAttention(128, 64, args.head3, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


class SemAttention2LN(nn.Module):
    def __init__(self, args):
        super(SemAttention2LN, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 128, args.head2, d_ff=args.t_dff, LN=True)
        self.layer3 = BasicSemAttention(128, 64, args.head3, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs

'''
语义信息的维度不变进行变换
'''
class SemAttention3(nn.Module):
    def __init__(self, args):
        super(SemAttention3, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


class SemAttention3_1(nn.Module):
    def __init__(self, args):
        super(SemAttention3_1, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, _ = self.layer2(outputs, forget)
        outputs, _ = self.layer3(outputs, outputs)
        
        return outputs


class SemAttention3_2(nn.Module):
    def __init__(self, args):
        super(SemAttention3_2, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention2(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


class SemAttention3_3(nn.Module):
    def __init__(self, args):
        super(SemAttention3_3, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, _ = self.layer1(outputs, forget)
        outputs, _ = self.layer2(outputs, outputs)
        outputs, _ = self.layer3(outputs, outputs)
        
        return outputs


class SemAttention3_2LN(nn.Module):
    def __init__(self, args):
        super(SemAttention3_2LN, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention2(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


class SemAttention3LN(nn.Module):
    def __init__(self, args):
        super(SemAttention3LN, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs

class SemAttention3LNALL(nn.Module):
    def __init__(self, args):
        super(SemAttention3LNALL, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff, LN=True)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs



class SemAttention3TransELN(nn.Module):
    def __init__(self, args):
        super(SemAttention3TransELN, self).__init__()
        self.args = args
        self.layer1 = BasicStageAttention(512, 128, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(128, 128, args.head1, d_ff=args.t_dff, LN=True)
        self.layer3 = BasicSemAttention(128, 32, args.head2, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


'''
把语义信息提取模块变为NLPStageAttention
'''
class SemAttention4(nn.Module):
    def __init__(self, args):
        super(SemAttention4, self).__init__()
        self.args = args
        self.layer1 = NLPStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs


class SemAttention4LN(nn.Module):
    def __init__(self, args):
        super(SemAttention4LN, self).__init__()
        self.args = args
        self.layer1 = NLPStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        
        return outputs


'''
加强版SemAttention4，对标SemAttention3，提高计算量，让区别仅存在于语义信息提取
'''
class SemAttention5(nn.Module):
    def __init__(self, args):
        super(SemAttention5, self).__init__()
        self.args = args
        self.layer1 = NLPStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


class SemAttention5LN(nn.Module):
    def __init__(self, args):
        super(SemAttention5LN, self).__init__()
        self.args = args
        self.layer1 = NLPStageAttention(1024, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer2 = BasicSemAttention(256, 256, args.head1, d_ff=args.t_dff, LN=True)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff, LN=False)
        
    def forward(self, inputs):
        outputs = inputs
        forget = inputs
        
        outputs, forget = self.layer1(outputs, forget)
        outputs, forget = self.layer2(outputs, forget)
        outputs, forget = self.layer3(outputs, forget)
        
        return outputs


class SemAttention6(nn.Module):
    def __init__(self, args):
        super(SemAttention6, self).__init__()
        self.args = args
        self.layer1 = NLPStageAttention(1024, 256, args.head1, d_ff=args.t_dff)
        self.layer2 = SemUpdate(256, 1024)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        
        outputs, _ = self.layer1(outputs, outputs)
        outputs = self.layer2(outputs, inputs)
        outputs, _ = self.layer3(outputs, outputs)
        
        return outputs


class SemAttention7(nn.Module):
    def __init__(self, args):
        super(SemAttention7, self).__init__()
        self.args = args
        self.layer1 = LGSemAttention(1024, 256, args.token1, args.head1, d_ff=args.t_dff)
        self.layer2 = SemUpdate(256, 1024)
        self.layer3 = BasicSemAttention(256, 64, args.head2, d_ff=args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        
        outputs, _ = self.layer1(outputs, outputs)
        outputs = self.layer2(outputs, inputs)
        outputs, _ = self.layer3(outputs, outputs)
        
        return outputs


class SemAttention8(nn.Module):
    def __init__(self, args):
        super(SemAttention8, self).__init__()
        self.args = args
        self.layer1 = LGSemAttention(1024, 256, args.token1, args.head1, d_ff=args.t_dff)
        self.layer2 = SemUpdate(256, 1024)
        self.layer3 = LowDimGenerate(256, 1024, 64, args.t_dff)
        
    def forward(self, inputs):
        outputs = inputs
        
        outputs, _ = self.layer1(outputs, outputs)
        outputs = self.layer2(outputs, inputs)
        outputs = self.layer3(outputs, inputs)
        
        return outputs