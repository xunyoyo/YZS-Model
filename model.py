import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum
from torch.nn import LSTM
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool

def exists(val):
    """Check if a value is not None."""
    return val is not None

def default(val, d):
    """Return the value if it exists, otherwise return the default."""
    return val if exists(val) else d

class Norm(nn.Module):
    """Normalization layer that applies Layer Normalization followed by a function."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # Normalization layer
        self.fn = fn  # Function to apply after normalization

    def forward(self, x, **kwargs):
        """Normalize the input then apply a function."""
        return self.fn(self.norm(x), **kwargs)

class FFN(nn.Module):
    """Feed-Forward Network that applies two linear transformations with a GELU activation in between."""
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
        """Pass the input through the feed-forward network."""
        return self.net(x)

class MSA(nn.Module):
    """Multihead Self-Attention module."""
    def __init__(self, dim, heads=5, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # Calculate the total dimension of all heads
        self.heads = heads
        self.scale = dim_head ** -0.5  # Scaling factor to normalize the dot products
        self.attend = nn.Softmax(dim=-1)  # Softmax to compute attention weights
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # Linear transformation for queries
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # Linear transformation for keys and values
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        """Compute self-attention and return the transformed output."""
        b, n, _, h = *x.shape, self.heads  # Unpack dimensions and number of heads
        context = default(context, x)  # Default context is the input itself unless specified
        if kv_include_self:
            context = torch.cat((x, context), dim=1)  # Optionally include input in key/value computation

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """A transformer model that uses multihead self-attention and feed-forward networks."""
    def __init__(self, dim, depth=6, heads=8, dim_head=92, mlp_dim=256, dropout=0.25):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)  # Final normalization layer

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, MSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                Norm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """Apply layers of attention and feed-forward networks in sequence."""
        for attn, ff in self.layers:
            x = attn(x) + x  # Apply attention and add the input (residual connection)
            x = ff(x) + x  # Apply feed-forward network and add the input (residual connection)
        return self.norm(x)  # Apply final normalization

class YZS(nn.Module):
    """Complete model combining GCN, Transformer, and LSTM for processing graph data."""
    def __init__(self, num_features=92, dim=128, dropout=0.251903250716151, depth=6, heads=8):
        super(YZS, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.output = 1

        self.conv = GCNConv(num_features, dim)  # Graph Convolutional Network layer

        self.transformer = Transformer(dim)  # Transformer module

        self.lstm = LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True, bidirectional=True)  # BiLSTM layer

        self.fc = nn.Sequential(  # Fully connected layers for output
            nn.Linear(2 * dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        """Process graph data through each module and produce an output."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv(x, edge_index))  # Apply GCN and activation

        transformer_input = x.unsqueeze(0)
        transformer_out = self.transformer(transformer_input)  # Process with Transformer

        lstm_out, (hn, cn) = self.lstm(transformer_out)  # Process with LSTM

        lstm_out = lstm_out[-1, :, :]  # Take the last output for pooling
        graph_features = global_add_pool(lstm_out, batch)  # Pool graph features

        out = self.fc(graph_features)  # Final output layer

        return out

if __name__ == '__main__':
    YZS()  # Initialize and possibly test the YZS model
