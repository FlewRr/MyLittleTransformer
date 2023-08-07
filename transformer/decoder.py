import torch 
import torch.nn as nn
from attention import Scaled_Dot_Product_Attention
from multi_head_attention import MultiHeadAttention
from utils import PointwiseFFN, PositionalEncoding
from typing import Optional

class TransformerDecoderLayer(nn.Module):
    """Auxillary class made up of self-attention, encoder-decoder attention and feedforward network
        
    Args:
        emb_size: the number of expected features in the input (required)
        num_heads: the number of heads in the multi-head-attention models (required)
        ffn_size: the dimension of feedforward network (required)
        dropout: the dropout value (default=0.1)
        attention_dropout: the dropout value used in the attention models (default=0.1)
        batch_first: If Tru, then the input and output tensors are provided
        as [batch, seq, feature]. Default: ``False`` [seq, batch, feature].
   
    """
    def __init__(self, emb_size, num_heads, ffn_size, dropout=0.1, attention_dropout=0.1, batch_first=False):
        super().__init__()  
        self.attention = MultiHeadAttention(emb_size, num_heads, attention_dropout, batch_first)
        self.attn_norm = nn.LayerNorm(emb_size)

        self.ffn = PointwiseFFN(emb_size, ffn_size, dropout)
        self.ffn_norm = nn.LayerNorm(emb_size)

    def forward(self, 
                dec_input : torch.Tensor,
                enc_output : torch.Tensor,
                dec_mask : Optional[torch.Tensor] = None,
                enc_mask : Optional[torch.Tensor] =None
                ):
        """Pass the inputs (and mask) through the decoder layer

        Args:
            dec_input: the sequence to the decoder layer (required)
            enc_output: the sequence from the last layer of encoder (required)
            dec_mask: the mask for target sequence (dec_input) (optional)
            enc_mask: the mask for enc_output sequence (optional)
        
        Shape:
            dec_input: [T, N, E]
            enc_output: [S, N, E]
            dec_mask: [N, T, T] or [1, T, T]
            enc_mask: [N, S, S] or [1, S, S]
        """

        dec_residual = dec_input
        decoder_attention, _ = self.attention(dec_input, dec_input, dec_input, dec_mask) # [T, N, E] --> [T, N, E]
        normed_attn_out = self.attn_norm(decoder_attention + dec_residual) 

        dec_residual = normed_attn_out
        attn_out, _ = self.attention(normed_attn_out, enc_output, enc_output, enc_mask) # --> [T, N, E]
        normed_attn_out = self.attn_norm(attn_out + dec_residual)

        ffn_residual = normed_attn_out
        ffn_out = self.ffn(normed_attn_out) # [T, N, E] --> [T, N, E]
        normed_ffn_out = self.ffn_norm(ffn_out + ffn_residual)

        return normed_ffn_out # [T, N, E]
    

class TransformerDecoder(nn.Module):
    def __init__(self, out_voc, emb_size, hidden_dim, num_heads, pad_idx=None, dropout=0.1, n_position=200, n_layers=6):
        super().__init__()

        self.emb_out = nn.Embedding(len(out_voc), emb_size, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(emb_size, n_position, dropout)
        self.out_voc = out_voc
        self.emb_size = emb_size
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(p=dropout)

        self.dec = nn.ModuleList([
            TransformerDecoderLayer(emb_size, num_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])


    def forward(self, 
                target_seq, 
                target_mask, 
                enc_output, 
                src_mask):
        
        
        dec_output = self.emb_out(target_seq)
        dec_output = self.positional_encoding(dec_output)
        dec_output = self.layer_norm(self.dropout(dec_output))

        for dec_layer in self.dec:
            dec_output = dec_layer(
                dec_input=dec_output, 
                enc_output=enc_output, 
                dec_mask=target_mask, 
                enc_mask=src_mask)  
        
        return dec_output
    

# if __name__ == "__main__":
#     dec = TransformerDecoder([1], 1, 1, 1, 0)