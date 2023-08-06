import torch
import torch.nn as nn

class PointwiseFFN(nn.Module):
    def __init__(self, inp_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim,inp_dim)
        self.layer_norm = nn.LayerNorm(inp_dim, eps=1e-6)
        self.dropout = dropout
    
    
    def forward(self, x):
        residual = x
        x = self.relu(self.linear1(x))
        x = self.dropout(self.linear2(x)) + residual
        
        return self.layer_norm(x)