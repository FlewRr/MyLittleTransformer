from transformer import attention, decoder, encoder, multi_head_attention, transformer
import torch
import torch.nn as nn
import torch.nn.functional as F



"""comment here to use

#___________________TESTS FOR ATTENTION___________________ 
batch_size = 9
query_size = 16
t_size = 7
keys_size = 16
value_size = 12

query = torch.randn((batch_size, query_size))
keys = torch.randn((t_size, batch_size, keys_size))
value = torch.randn((t_size, batch_size, value_size))

dpa = attention.Scaled_Dot_Product_Attention(query_size)

attention = F.scaled_dot_product_attention(query, keys, value)
attention2 = dpa(query, keys, value)

print(attention.shape == attention2[0].shape)
print(torch.equal(attention, attention2[0]))
#_________________________________________________________

comment here to use """
