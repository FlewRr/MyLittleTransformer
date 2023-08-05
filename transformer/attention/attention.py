import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, query_size : torch.int32):
        super().__init__()
        self.query_size = query_size
        self.scale = 1 / math.sqrt(query_size)
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, query : torch.float32, keys : torch.float32, value : torch.float32, mask : torch.BoolTensor = None):
        """
        Computes attention response and weights
        :param query: float32 [B, Q]
        :param key: float32 [T, B, K]
        :param value: float32 [T, B, V]
        :assuming K == Q
        :returns: output_attention[batch_size, enc_size], attention_probs[batch_size, ninp]
            - output_attention - attention response vector i.e. scaled dot-product attention
            - attention_probs - attention weights after softmax
        """
        
        attention = query @ keys.transpose(-2, -1) # [B, Q] * [B, K, T] --> [T, B, B]
        
        if mask is not None:
            attention.masked_fill(not mask, -1e9) # apply mask to the attention tensor
        
        attention_probs = self.softmax(self.scale * attention)
        output_attention = attention_probs @ value # [T, B, B] * [T, B, V] --> [T, B, V]
        
        return output_attention, attention_probs