import torch
import torch.nn as nn
import math


class PointwiseFFN(nn.Module):
    def __init__(self, inp_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim,inp_dim)
        self.layer_norm = nn.LayerNorm(inp_dim, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
    
    
    def forward(self, x):
        residual = x
        x = self.relu(self.linear1(x))
        x = self.dropout(self.linear2(x)) + residual
        
        return self.layer_norm(x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, n_position=200, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(n_position).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pos_table = torch.zeros(n_position, 1, hidden_dim)
        pos_table[:, 0, 0::2] = torch.sin(position * div_term)
        pos_table[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pos_table', pos_table)

    def forward(self, x):
        return self.dropout(x + self.pos_table[x.size(0)])
    


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    _, len = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len, len), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

# if __name__ == "__main__":
#     pe = PostitionalEncoding(4, 200)
#     x = torch.randn((11, 19, 4))
#     print(pe(x).shape)


"""
TO DO:
1. bit_mask - function to create bit mask for padding
2. bit_dec_mask - function to create bit mask for decoder
3. come up with better names for functions above
"""