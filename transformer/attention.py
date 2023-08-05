import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, query_size, dropout=0.0):
        super().__init__()
        self.query_size = query_size
        self.scale = 1 / math.sqrt(query_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = dropout
    
    def forward(self, 
                query : torch.Tensor,
                key : torch.Tensor, 
                value : torch.float32, 
                mask : Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes attention response and weights
        :param query: the query of the attention function, float32 [B, Q]
        :param key: the keys of the attention function, float32 [T, B, K]
        :param value: the values of the attention function, float32 [T, B, V]
        :assuming K == Q
        :returns: output_attention[batch_size, enc_size], attention_probs[batch_size, ninp]
            - output_attention - attention response vector i.e. scaled dot-product attention
            - attention_probs - attention weights after softmax
        """
        
        attention = query @ key.transpose(-2, -1) # [B, Q] * [B, K, T] --> [T, B, B]
        
        if mask is not None:
            attention.masked_fill(not mask, -1e9) # apply mask to the attention tensor
        
        attention_probs = F.dropout(self.softmax(self.scale * attention), p=self.dropout)
        output_attention = attention_probs @ value # [T, B, B] * [T, B, V] --> [T, B, V]
        
        return output_attention, attention_probs