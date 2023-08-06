import torch 
import torch.nn as nn
from attention import Scaled_Dot_Product_Attention
from multi_head_attention import MultiHeadAttention
from utils import PointwiseFFN


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, ffn_size, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, attention_dropout)
        self.attn_norm = nn.LayerNorm(emb_size)

        self.ffn = PointwiseFFN(emb_size, ffn_size, dropout)
        self.ffn_norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        residual = x
        attention_out, _ = self.attention(x, x, x) # [L, N, E] --> [L, N, E]
        normed_attention_out = self.attn_norm(attention_out + residual)
        
        ffn_residual = normed_attention_out
        ffn_out = self.ffn(normed_attention_out)
        normed_ffn_out = self.ffn_norm(ffn_out + ffn_residual)
        
        return normed_ffn_out