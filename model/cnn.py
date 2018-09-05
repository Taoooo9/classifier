import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, embedding, embedding_num, embedding_size):
        super(CNN, self).__init__()
        input_channel = 1
        class_num = 5
        kernel_size = (3, 4, 5)
        kernel_num = 100
        data_dim = 300
        self.embed = nn.Embedding(embedding_num, data_dim)
        self.embed.weight.data.copy_(torch.from_numpy(embedding))
        self.conv2d = [nn.Conv2d(in_channels = input_channel, out_channels = kernel_num, kernel_size = (k, data_dim),
                                stride = (1, 1), padding = (k // 2, 0), dilation = 1) for k in kernel_size]
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(len(kernel_size) * kernel_num, class_num)


    def forward(self, x):
        x = self.embed(x)  # (batch_size, max_length, embedding_dim)
        x = x.unsqueeze(1)  # (b, c_in, len, dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv2d]  # (b, c_in, L)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (b, c_out)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.linear(x)  # (len(Ks)*Co, class_num)
        return logit



