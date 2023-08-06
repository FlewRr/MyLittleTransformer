import  decoder
import encoder
import multi_head_attention 
import attention
import torch
import torch.nn as nn
import torch.nn.functional as F



"""comment here to use

#___________________TESTS FOR ATTENTION___________________ 
batch_size = 9
query_size = 16
source_size = 8
target_size = 10
keys_size = 16
value_size = 1#

query = torch.randn((target_size, batch_size, query_size))
keys = torch.randn((source_size, batch_size, keys_size))
value = torch.randn((source_size, batch_size, keys_size))
mask = torch.randn((batch_size, target_size, source_size)).uniform_() > 0.8

k = torch.randn((batch_size, target_size, source_size))
k = torch.masked_fill(k, mask == 0, 0)

dpa = attention.Scaled_Dot_Product_Attention(query_size)

attention2 = dpa(query, keys, value, mask)

print(attention2[0].shape)

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