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
                attn_mask : Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes attention response and weights
        :param query: the query of the attention function, float32 [T, B, E]
        :param key: the keys of the attention function, float32 [S, B, E]
        :param value: the values of the attention function, float32 [S, B, E]
        :returns: output_attention[batch_size, enc_size], attention_probs[batch_size, ninp]
            - output_attention - attention response vector i.e. scaled dot-product attention
            - attention_probs - attention weights after softmax
        """
        
        assert query.size(-1) == key.size(-1) == value.size(-1), "The feature dim of query, key, value must be equal."
        assert key.size() == value.size(), "Shape of key, value must match"

        src_len, tgt_len = key.size(-3), query.size(-3)
        batch_heads = max(query.size(-2), key.size(-2))
        if attn_mask is not None:
            if attn_mask.dim() != 3:
                raise RuntimeError("attn_mask must be a 3D tensor.")
            if (
                (attn_mask.size(-1) != src_len)
                or (attn_mask.size(-2) != tgt_len)
                or (attn_mask.size(-3) != 1 and attn_mask.size(-3) != batch_heads)
            ):
                raise RuntimeError("The size of the attn_mask is not correct.")
            if attn_mask.dtype != torch.bool:
                raise RuntimeError("Only bool tensor is supported for attn_mask")

        query, key, value = query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3)
        
        attention = query @ key.transpose(-2, -1) # [S, B, E] * [T, E, B] --> [B, T, S]
    
        if attn_mask is not None: 
            attention = torch.masked_fill(attention, attn_mask == 0, -1e9) # apply mask to the attention tensor
        
        attention_probs = F.dropout(self.softmax(self.scale * attention), p=self.dropout)
        output_attention = attention_probs @ value # [T, B, B] * [T, B, V] --> [T, B, V]
        
        return output_attention, attention_probs