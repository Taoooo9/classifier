import torch
import torch.nn as nn
import torch.nn.functional as F


class RRN(nn.Module):

    def __init__(self, args,  data_num, data_dim, padding, embedding):
        super(RRN, self).__init__()
        self.embed = nn.Embedding(data_num, args.embed_dim, padding_idx = padding)
        self.embed.weight.data.copy_(torch.from_numpy(embedding))
        self.rnn = nn.RNN(input_size = args.embed_dim, hidden_size = args.hidden_size,
                          dropout = args.dropout_gru,
                          nonlinearity = 'relu',
                          batch_first = True, bidirectional = True)
        self.dropout = nn.Dropout(args.dropout_rnn)
        self.linear = nn.Linear(args.embed_dim, data_dim)


    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = F.tanh(torch.transpose(x, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        logit = self.linear(x)
        return logit


