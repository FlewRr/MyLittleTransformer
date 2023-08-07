import torch 
import torch.nn as nn
from attention import Scaled_Dot_Product_Attention
from multi_head_attention import MultiHeadAttention
from utils import PointwiseFFN, PositionalEncoding
from typing import Optional

class TransformerEncoderLayer(nn.Module):
    """Auxillary class made up of self-attention and feedforward network
        
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
        self.attention = MultiHeadAttention(emb_size, num_heads, attention_dropout, batch_first=batch_first)
        self.attn_norm = nn.LayerNorm(emb_size)

        self.ffn = PointwiseFFN(emb_size, ffn_size, dropout)
        self.ffn_norm = nn.LayerNorm(emb_size)

    def forward(self, 
                source : torch.Tensor, 
                attn_mask : Optional[torch.Tensor] = None
                ):
        """Pass the input through encoder layer

        Args:
            source: the sequence to the encoder layer (required)
            attn_mask: the mask for the source sequen (optional)

        Shape:
            source: [S, E] for unbatched input, [S, N, E] for batched (batch_first=False), [N, S, E] (batch_first=True) 
            attn_mask: [1, S, S] or [N, S, S]
        """
        residual = source
        attention_out, _ = self.attention(source, source, source, attn_mask) # [S, N, E] --> [S, N, E]
        normed_attention_out = self.attn_norm(attention_out + residual)
        # enc_self_attention = normed_attention_out

        ffn_residual = normed_attention_out
        ffn_out = self.ffn(normed_attention_out) # [S, N, E] --> [S, N, E]
        normed_ffn_out = self.ffn_norm(ffn_out + ffn_residual)
        
        return normed_ffn_out # [S, N, E]
    

class TransformerEncoder(nn.Module):
    """TransformerEncoder is the stack of N Encoder Layers.

    Args:
        inp_voc: vocabulary used to get embedding of source sequence (required)
        emb_size: the number of expected features in the input (required)
        hidden_dim: the number of dimensions used in the feedforward network inside encoder layer (required)
        num_heads: the number of heads used in the multi-head-attention models (required)
        pad_idx: padding mask for embeddings (required)
        dropout: dropout parameter (default=0.1)
        n_position: number of position used in the positional encoding (default=200)
        n_layers: number of layers (default=6)
    """
    def __init__(self, inp_voc, emb_size,  hidden_dim, num_heads, pad_idx=None, dropout=0.1, n_position=200, n_layers=6):
        super().__init__()  
        self.emb_inp = nn.Embedding(len(inp_voc), emb_size, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(emb_size, n_position, dropout)
        self.inp_voc = inp_voc
        self.emb_size = emb_size
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(p=dropout)

        self.enc = nn.ModuleList([
            TransformerEncoderLayer(emb_size, num_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

    
    def forward(self, 
                input : torch.Tensor, 
                src_mask : Optional[torch.Tensor] = None):
        """Pass the input through the encoder layers in turn.
        
        Args:
            input: the sequence to the encoder (required)
            src_mask: the mask for the source sequence (optional)

        Shape:  
            input: [S, O] for unbatched input, [S, N, O] for batched (batch_first=False), [N, S, O] (batch_first=True).
            src_mask: [1, S, S] or [N, S, S]
        """

        embed = self.positional_encoding(self.emb_inp(input)) # [..,, O] --> [..., E]
        embed = self.layer_norm(self.dropout(embed))
        
        encoder_state = embed

        for encoder_layer in self.enc:
            encoder_state = encoder_layer(encoder_state, src_mask)

        return encoder_state # [S, N, E]
    

# if __name__ == "__main__":
#     enc = TransformerEncoder([1], 1, 0, 1, 1)
