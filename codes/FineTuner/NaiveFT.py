import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class MLPFineTuning(nn.Module):
    def __init__(self, target_dim, sem_dim):
        super(MLPFineTuning, self).__init__()
        self.target_dim = target_dim
        self.sem_dim = sem_dim
        
        self.FTlayer = nn.Linear(target_dim+sem_dim, target_dim)
    
    def forward(self, sem, embedding):
        output = self.FTlayer(torch.cat((sem, embedding), dim=-1))
        if embedding.shape==output.shape:
            output=output+embedding
        else:
            output=output
        
        return output