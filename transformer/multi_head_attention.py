import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, odel_size, query_size, key_size, value_size):
        # matrices [n_heads, model_size, (query, key, value)_size]
        
