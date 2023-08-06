import torch 
import torch.nn as nn
from attention import Scaled_Dot_Product_Attention
from multi_head_attention import MultiHeadAttention
from utils import PointwiseFFN, PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, ffn_size, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, attention_dropout)
        self.attn_norm = nn.LayerNorm(emb_size)

        self.ffn = PointwiseFFN(emb_size, ffn_size, dropout)
        self.ffn_norm = nn.LayerNorm(emb_size)

    def forward(self, x, attn_mask=None):
        residual = x
        attention_out, _ = self.attention(x, x, x, attn_mask) # [L, N, E] --> [L, N, E]
        normed_attention_out = self.attn_norm(attention_out + residual)
        enc_self_attention = normed_attention_out # to be returned

        ffn_residual = normed_attention_out
        ffn_out = self.ffn(normed_attention_out)
        normed_ffn_out = self.ffn_norm(ffn_out + ffn_residual)
        
        return normed_ffn_out, enc_self_attention
    

class TransformerEncoder(nn.Module):
    def __init__(self, inp_voc, emb_size, pad_idx, hidden_dim, num_heads, dropout=0.1, n_position=200, n_layers=6):
        super().__init__()  
        self.emb_inp = nn.Embedding(len(inp_voc), emb_size, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(hidden_dim, n_position, dropout)
        self.inp_voc = inp_voc
        self.emb_size = emb_size
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(p=dropout)

        self.enc = nn.ModuleList([
            TransformerEncoderLayer(emb_size, num_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

    
    def forward(self, input, src_mask=None):

        embed = self.positional_encoding(self.emb_inp(input))
        embed = self.layer_norm(self.dropout(embed))
        
        encoder_state = embed 

        for encoder_layer in self.enc:
            encoder_state = encoder_layer(encoder_state)

        return encoder_state

    
    def encode(self, input, **flags):

        embed = self.positional_encoding(self.emb_inp(input))
        embed = self.layer_norm(self.dropout(embed))
        
        encoder_state = embed 

        for encoder_layer in self.enc:
            encoder_state = encoder_layer(encoder_state)

        return encoder_state 
    

# if __name__ == "__main__":
#     enc = TransformerEncoder([1], 1, 0, 1, 1)
