import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm

class MYMODEL(torch.nn.Module):
    def __init__(self, model_output_dim=1, dropout=0.2):
        super(MYMODEL, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        


    def forward(self, data):
        pass