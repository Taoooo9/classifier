import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, args, embedding_size, embedding_dim, padding_id, embedding):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(embedding_size, args.embed_dim, padding_idx = padding_id)
        self.embed.weight.data.copy_(torch.from_numpy(embedding))
        self.dropout = nn.Dropout(args.dropout_embed)
        self.bilstm = nn.LSTM(args.embed_dim, args.hidden_size, dropout = args.dropout_rnn, batch_first = True, bidirectional = True)
        self.linear = nn.Linear(args.hidden_size * 2, embedding_dim)


    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        x, _ = self.bilstm(x)

        x = F.tanh(torch.transpose(x, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.linear(x)
        return x