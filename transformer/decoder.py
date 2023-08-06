import torch 
import torch.nn as nn
from attention import Scaled_Dot_Product_Attention
from multi_head_attention import MultiHeadAttention
from utils import PointwiseFFN

class TransformerDecoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, ffn_size, dropout=0.1, attention_dropout=0.1):
        super().__init__()  
        self.attention = MultiHeadAttention(emb_size, num_heads, attention_dropout)
        self.attn_norm = nn.LayerNorm(emb_size)

        self.ffn = PointwiseFFN(emb_size, ffn_size, dropout)
        self.ffn_norm = nn.LayerNorm(emb_size)

    def forward(self, 
                dec_input,
                enc_output,
                dec_mask=None,
                enc_mask=None
                ):
        
        dec_residual = dec_input
        decoder_attention, _ = self.attention(dec_input, dec_input, dec_input, dec_mask)
        normed_attn_out = self.attn_norm(decoder_attention + dec_residual)

        dec_residual = normed_attn_out
        attn_out, _ = self.attention(normed_attn_out, enc_output, enc_output, enc_mask)
        normed_attn_out = self.attn_norm(attn_out + dec_residual)

        ffn_residual = normed_attn_out
        ffn_out = self.ffn(normed_attn_out)
        normed_ffn_out = self.ffn_norm(ffn_out + ffn_residual)

        return normed_ffn_out