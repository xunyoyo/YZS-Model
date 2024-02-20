"""
By xunyoyo & kesmeey
Part of code come from GitHub:
https://github.com/ziduzidu/CSDTI
https://github.com/ltorres97/FS-CrossTR
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum
from torch.nn import LSTM
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MSA(nn.Module):
    def __init__(self, dim, heads=5, dim_head=64, dropout=0.):
        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim=1)  # CA needs the cls token to exchange information

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth=5, heads=8, dim_head=24, mlp_dim=256, dropout=0.2):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, MSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                Norm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MYMODEL(torch.nn.Module):
    def __init__(self, num_features=24, dim=48, dropout=0.2):
        super(MYMODEL, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.output = 1

        self.conv11 = GCNConv(num_features, dim)
        self.conv12 = GCNConv(num_features, dim)

        self.conv21 = GCNConv(dim * 2, dim)
        self.conv22 = GCNConv(dim * 2, dim)

        self.transformer = Transformer(dim * 2)

        self.lstm = LSTM(input_size=dim * 2, hidden_size=dim, num_layers=1, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index2 = data.edge_index2

        x1 = self.relu(self.conv11(x, edge_index))
        x2 = self.relu(self.conv12(x, edge_index2))

        x12 = torch.cat((x1, x2), dim=1)

        x1 = self.relu(self.conv21(x12, edge_index))
        x2 = self.relu(self.conv22(x12, edge_index2))

        x12 = torch.cat((x1, x2), dim=1)

        transformer_input = x12.unsqueeze(0)
        transformer_out = self.transformer(transformer_input)

        lstm_out, (hn, cn) = self.lstm(transformer_out)

        lstm_out = lstm_out[-1, :, :]
        graph_features = global_mean_pool(lstm_out, batch)

        out = self.fc(graph_features)

        return out


if __name__ == '__main__':
    MYMODEL()
