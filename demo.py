import torch
import torch.nn as nn


embedding = nn.Embedding(10, 3, padding_idx=5)
input = torch.LongTensor([[0,2,1,5]])
print(embedding(input))
# tensor([[[ 0.0000,  0.0000,  0.0000],
#          [ 0.1535, -2.0309,  0.9315],
#          [ 0.0000,  0.0000,  0.0000],
#          [-0.1655,  0.9897,  0.0635]]])