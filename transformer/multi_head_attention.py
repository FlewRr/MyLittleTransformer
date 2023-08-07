import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.0, batch_first=False, InputProjectContainer = None, 
                output_proj=None, attention_layer=None):
        # matrices [n_heads, model_size, (query, key, value)_size]
        super().__init__()

        self.embed_dim = embed_dim
        self.nhead = nhead
        self.dropout = dropout
        self.batch_first = batch_first

        # linear -- input_proj
        # attention layer -- scaled dot product attention
        # linear -- output_proj
        
        self.attention_layer = attention_layer

        if attention_layer is None:
            self.attention_layer = attention.Scaled_Dot_Product_Attention(embed_dim, dropout)

        if InputProjectContainer:
            self.query_proj, self.key_proj, self.value_proj = InputProjectContainer
        else:   
            self.query_proj, self.key_proj, self.value_proj = nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim)

        if output_proj:
            self.output_proj = output_proj
        else:
            self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, 
                query : torch.Tensor, 
                key : torch.Tensor, 
                value : torch.Tensor,
                attn_mask : Optional[torch.Tensor] = None):

        """
        Args:
            query : The query of the attention funtion - (Tensor)
            key : The key of the attention function - (Tensor)
            value : The value of the attention function - (Tensor)
            attn_mask : 3D mask that prevents attention to certain positions - (BoolTensor, optional)
        Shape:
            -Inputs:
                - query : [T, N, E]
                - key : [S, N, E]
                - value : [S, N, E]
            -Outputs:
                - attention_output : [T, N, E] or [N, T, E] if batch_first=True
                - attention_output_weights (probs) : [N * H, T, S]
        """

        if self.batch_first:
            query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)

        tgt_len, src_len, bsz, embed_dim = query.size(-3), key.size(-3), query.size(-2), query.size(-1)

        query_proj, key_proj, value_proj = self.query_proj(query), self.key_proj(key), self.value_proj(value)
        
        assert embed_dim % self.nhead == 0, "embed_dim must be divisible by the number of heads"
        head_dim = embed_dim // self.nhead
        
        query_proj = query_proj.reshape(tgt_len, bsz * self.nhead, head_dim) # [T, N * n_heads, E // n_heads]
        key_proj = key_proj.reshape(src_len, bsz * self.nhead, head_dim) # [S, N * n_heads, E // n_heads]
        value_proj = value_proj.reshape(src_len, bsz * self.nhead, head_dim) # [S, N * n_heads, E // n_heads]]
    

        attention_output, attention_output_weigths = self.attention_layer(
            query_proj, key_proj, value_proj, attn_mask) # [N * n_heads, T, S]
        attention_output = attention_output.reshape(tgt_len, bsz, embed_dim) # [T, N, E]
        attention_output = self.output_proj(attention_output) # [T, N, E]

        if self.batch_first:
            attention_output = attention_output.transpose(-3, -2)  # [N, T, E]

        return attention_output, attention_output_weigths