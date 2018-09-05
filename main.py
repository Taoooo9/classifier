import pickle
import torch
import torch.cuda
from seed import seed
from configer import config
from dataloader.dataloading import *
from vocab.vocabulary import *
from vocab.embedding import *
from model.lstm import *
from model.cnn import *
from model.gru import *
from model.r_nn import *
from driver.train import *

if __name__ == '__main__':

    # random
    seed.seed(888)

    # parameters
    config = config.default()

    # read data
    train_data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)

    # create vocab
    src_vocab, target_vocab = create_vocab(train_data, config.vocab_size)
    pickle.dump(src_vocab, open(config.save_src_vocab_path, 'wb'))
    pickle.dump(target_vocab, open(config.save_tgt_vocab_path, 'wb'))

    #  embedding
    embedding = src_embedding(src_vocab, config.embedding_file)

    # model
    if config.which_model == 'lstm':
        model = LSTM(config, src_vocab.size, target_vocab.size, PAD, embedding)
    elif config.which_model == 'gru':
        model = GRU(config, src_vocab.size, target_vocab.size, PAD, embedding)
    elif config.which_model == 'rnn':
        model = RRN(config, src_vocab.size, target_vocab.size, PAD, embedding)
    elif config.which_model == 'cnn':
        model = CNN(embedding, src_vocab.size, target_vocab.size)
    else:
        raise RuntimeError("Invalid optim method: " + config.which_model)

    train(model, train_data, dev_data, test_data, src_vocab, target_vocab, config)







