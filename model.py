"""
By xunyoyo & kesmeey
Part of code come from GitHub:
https://github.com/ziduzidu/CSDTI
https://github.com/ltorres97/FS-CrossTR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv
from torch.nn import LSTM

from torch.nn.modules.batchnorm import _BatchNorm


class Transformer(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class MYMODEL(torch.nn.Module):
    def __init__(self, num_features=17, dim=32, dropout=0.2):
        super(MYMODEL, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.output = 1

        self.conv11 = GCNConv(num_features, dim)
        self.conv12 = GCNConv(num_features, dim)

        self.conv21 = GCNConv(dim * 2, dim)
        self.conv22 = GCNConv(dim * 2, dim)

        self.lstm = LSTM(input_size=dim * 2, hidden_size=dim, num_layers=1, batch_first=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index2 = data.edge_index2

        x1 = self.relu(self.conv11(x, edge_index))
        x2 = self.relu(self.conv12(x, edge_index2))

        x12 = torch.cat((x1, x2), dim=1)

        x1 = self.relu(self.conv21(x12, edge_index))
        x2 = self.relu(self.conv22(x12, edge_index2))

        x12 = torch.cat((x1, x2), dim=1)

        lstm_input = x12.unsqueeze(0)
        lstm_out, (hn, cn) = self.lstm(lstm_input)
