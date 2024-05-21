import sys
import os
import logging
import torch
import math
import torch.nn as nn
import numpy as np

class Constant(nn.Module):
    def __init__(self):
        super(Constant, self).__init__()
        logging.info('Init Constant Pruner')
   
    def forward(self, inputs):
        return inputs
