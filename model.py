import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum
from torch.nn import LSTM
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool

class Norm(nn.Module):
    """
    Normalization layer that applies Layer Normalization and then a function.
    """
    def __init__(self, dim, fn):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(dim)  # Apply Layer Normalization
        self.fn = fn  # Function to apply after normalization

    def forward(self, x, **kwargs):
        # Normalize and apply function
        return self.fn(self.norm(x), **kwargs)

class FeedForwardNetwork(nn.Module):
    """
    A feed-forward neural network with GELU non-linearity and dropout.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForwardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # GELU activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pass input through the network
        return self.net(x)

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, dim, heads=5, dim_head=64, dropout=0.):
        super(MultiHeadSelfAttention, self).__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5  # Scaling factor for dot products
        self.attend = nn.Softmax(dim=-1)  # Softmax for attention
        self.dropout = nn.Dropout(dropout)  # Dropout for attention
        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # Query transformation
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # Key and value transformations
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = context if context is not None else x
        if kv_include_self:
            context = torch.cat((x, context), dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    """
    A block of transformer layers consisting of multi-head self-attention and feed-forward networks.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([Norm(dim, MultiHeadSelfAttention(dim, heads, dim_head, dropout)),
                           Norm(dim, FeedForwardNetwork(dim, mlp_dim, dropout))])
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # Apply attention and add residual
            x = ff(x) + x  # Apply feed-forward and add residual
        return self.norm(x)

class MYMODEL(nn.Module):
    """
    A composite model combining graph convolution, transformer, and LSTM layers.
    """
    def __init__(self, num_features, dim, dropout, depth, heads, hidden_dim=256):
        super(MYMODEL, self).__init__()
        self.conv = GCNConv(num_features, dim)  # Graph Convolutional Network layer
        self.transformer = TransformerBlock(dim, depth, heads, dim, hidden_dim * 2, dropout)
        self.lstm = LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)  # LSTM layer
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Output layer
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv(x, edge_index).relu()  # Apply GCN and ReLU
        transformer_out = self.transformer(x.unsqueeze(0))  # Pass through transformer
        lstm_out, _ = self.lstm(transformer_out)  # Pass through LSTM
        graph_features = global_add_pool(lstm_out.squeeze(0), batch)  # Global pooling
        return self.fc(graph_features)  # Final fully connected layer

if __name__ == '__main__':
    model = MYMODEL(num_features=92, dim=128, dropout=0.25, depth=6, heads=8)
