import re
import numpy as np
import torch
from torch.autograd import Variable

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def read_corpus(data_file):
    data_list = []
    with open(data_file, encoding = 'utf-8') as f:
        for data_line in f:
            str = data_line.strip()
            if str == '' or len(str) == 0:
                print("an empty sentence, please check")
            str = str[0:str.index("|||") - 1]
            str = clean_str(str)
            str = str.split(' ')
            label = data_line[-2]
            data_list.append((str, label))
    return data_list


def create_batch(data, batch_size, shuffle = True):
    data_size = len(data)
    if shuffle:
        np.random.shuffle(data)

    src_ids = sorted(range(data_size), key=lambda src_id: len(data[src_id][0]), reverse=True)
    data = [data[src_id] for src_id in src_ids]

    unit = []
    instances = []
    for instance in data:
       instances.append(instance)
       if len(instances) == batch_size:
           unit.append(instances)
           instances = []

    if len(instances) > 0:
         unit.append(instances)

    for batch in unit:
        yield batch


def pair_data_variable(batch, src_vocab, target_vocab, args):
    batch_size = len(batch)
    src_length = [len(batch[idx][0]) for idx in range(batch_size)]
    max_len = int(src_length[0])

    src_words = Variable(torch.LongTensor(batch_size, max_len).zero_(), requires_grad=False)
    src_target = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)


    for idx, instance in enumerate(batch):
        sentence = src_vocab.word22id(instance[0])
        for idj, value in enumerate(sentence):
            src_words[idx][idj] = value
        src_target[idx] = target_vocab.word22id(instance[1])

    if args.use_cuda:
        src_words = src_words.cuda()
        src_target = src_target.cuda()

    return src_words, src_target, src_length





