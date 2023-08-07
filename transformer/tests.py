import  decoder
import encoder
import multi_head_attention 
import attention
import encoder
import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F



"""comment here to use

#___________________TESTS FOR ATTENTION___________________ 
batch_size = 64
embed_size=32
target_size=16
source_size=8

query = torch.randn((target_size, batch_size, embed_size))
keys = torch.randn((source_size, batch_size, embed_size))
value = torch.randn((source_size, batch_size, embed_size))
mask = torch.randn((batch_size, target_size, source_size)).uniform_() > 0.8

k = torch.randn((batch_size, target_size, source_size))
k = torch.masked_fill(k, mask == 0, 0)

dpa = attention.Scaled_Dot_Product_Attention(embed_size)

attention2 = dpa(query, keys, value, mask)

print(attention2[1].shape)

#_________________________________________________________

comment here to use """

""" comment here to use

#______________TESTS FOR MULTI_HEAD_ATTENTION_____________
batch_size = 32
target_size  = 16
embedding_size = 16
src_size = 8
nheads = 4

query = torch.randn((target_size, batch_size, embedding_size))
keys = torch.randn((src_size, batch_size, embedding_size))
value = torch.randn((src_size, batch_size, embedding_size))
mask = torch.randn((batch_size * nheads, target_size, src_size)).uniform_() > 0.8

mha = multi_head_attention.MultiHeadAttention(embedding_size, nheads)
mha_torch = nn.MultiheadAttention(embedding_size, nheads)
attention_output, attention_output_weights = mha(query, keys, value)
attention_output_torch, attention_output_weights_torch = mha_torch(query, keys, value)
print(attention_output_weights_torch)
print(attention_output_weights)
#_________________________________________________________

comment here to use"""



""" comment here to use

#______________TESTS FOR TRANSFORMER ENCODER_____________

batch_size = 1
target_size = 256
embedding_size = 64
nhead = 2
ffn_size = 128
x = torch.randn((batch_size, target_size, embedding_size))
e = encoder.TransformerEncoderLayer(embedding_size, nhead, ffn_size)

encoder_result = e(x)
print(x.shape == encoder_result.shape)
#_________________________________________________________

comment here to use"""


""" comment here to use

#______________TESTS FOR TRANSFORMER DECODER_____________

batch_size = 1
target_size = 256
embedding_size = 64
nhead = 2
ffn_size = 128
x = torch.randn((batch_size, target_size, embedding_size))
e = decoder.TransformerDecoderLayer(embedding_size, nhead, ffn_size)

decoder_result = e(x, encoder_result)
print(x.shape == decoder_result.shape)
#_________________________________________________________

comment here to use"""