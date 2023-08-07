import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import TransformerDecoder
from encoder import TransformerEncoder
from utils import get_pad_mask, get_subsequent_mask
from torch.nn.init import xavier_uniform_


class Transformer(nn.Module):
    def __init__(self, inp_voc, out_voc, enc_pad_idx=None, dec_pad_idx=None, emb_size=512, hidden_dim=2048, n_layers=6, dropout=0.1, n_position=200, num_heads=8):
        super().__init__()
        """A Transformer model.
        
        Args:
            inp_voc: source vocabulary (required)
            out_voc: target vocabulary (required)
            enc_pad_idx: padding mask for encoder (defalut=None).
            dec_pad_idx: padding mask for decoder (default=None).
            emb_size: the number of expected features in the encoder/decoder inputs (default=512).
            hidden_dim: the number of dimension used in feedforward network (default=2048).
            n_layers: the number of layers in the encoder/decoder (default=6).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            num_heads: number of heads of the multi-head-attention models (default=8).
        """
        self.enc_pad_idx, self.dec_pad_idx = enc_pad_idx, dec_pad_idx
        self.inp_voc, self.out_voc = inp_voc, out_voc

        self.encoder = TransformerEncoder(
            inp_voc=inp_voc, emb_size=emb_size, 
            pad_idx=enc_pad_idx, hidden_dim=hidden_dim, 
            num_heads=num_heads, dropout=dropout, 
            n_position=n_position, n_layers=n_layers
        )

        self.decoder = TransformerDecoder(
            out_voc=out_voc, emb_size=emb_size,
            hidden_dim=hidden_dim, num_heads=num_heads,
            pad_idx=dec_pad_idx, dropou=dropout,
            n_position=n_position, n_layers=n_layers
        )

        self._reset_parameters()

        self.emb_size = emb_size

        self.last = nn.Linear(emb_size, len(out_voc), bias=False)


    def forward(self, 
                source : torch.Tensor, 
                target : torch.Tensor
                ):
        """Take in and process masked source/target sequences.

        Args:
            source: the sequence to the encoder (required).
            target: the sequence to the decoder (required).
            
        Shape:
            source: [S, E] for unbatched input, [S, N, E] if batch_first = False, [N, S, E] if batch_first = True.
            target: [T, E] for unbatched input, [T, N, E] if batch_first = False, [N, T, E] if batch_first = True
        """

        src_mask = get_pad_mask(source, self.enc_pad_idx)
        target_mask = get_pad_mask(target, self.dec_pad_idx) & get_subsequent_mask(target)

        enc_output = self.encoder(source)
        dec_output = self.decoder(target, target_mask, enc_output, src_mask) # [T, N, E]
        
        logits = self.last(dec_output) # [T, N, OUT_VOC] -- logits for tokens in out vocabulary
        
        return logits.view(-1, logits.size(2)) # [T * N, OUT_VOC]
     

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)