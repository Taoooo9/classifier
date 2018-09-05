from collections import Counter

PAD, UNK = 0, 1
PAD_S, UNK_S = '<pad>', '<unk>'


def create_vocab(data, vocab_size):
    word_count = Counter()
    for unit in data:
        for word in unit[0]:
            word_count[word] += 1
    label_count = Counter()
    for unit in data:
        label_count[unit[1]] += 1

    most_word = [ite for ite, it in word_count.most_common(vocab_size)]
    most_label = [ite for ite, it in label_count.most_common(vocab_size)]

    src_vocab = SrcWord(most_word)
    target_vocab = SrcLabel(most_label)

    return src_vocab, target_vocab


class SrcWord:
    def __init__(self, most_word):
        self.extra_list = [PAD_S, UNK_S]
        self.id2word = self.extra_list + most_word
        self.word2id = {}
        for index, word in enumerate(self.id2word):
            self.word2id[word] = index
        if len(self.id2word) != len(self.word2id):
            print('Error!, please check data!')

    @property
    def size(self):
        return len(self.word2id)

    def word22id(self, xx):
        if isinstance(xx, list):
            return [self.word2id.get(word, UNK) for word in xx]
        return self.word2id[xx]

    def id22word(self, xx):
        if isinstance(xx, list):
            return [self.id2word[idx] for idx in xx]
        return self.id2word[xx]



class SrcLabel:
    def __init__(self, most_label):
        self.id2word = most_label
        self.word2id = {}
        for index, label in enumerate(self.id2word):
            self.word2id[label] = index
        if len(self.id2word) != len(self.word2id):
            print('Error!, please check data!')

    @property
    def size(self):
        return len(self.word2id)

    def word22id(self, xx):
        if isinstance(xx, list):
            return [self.word2id.get(word, UNK) for word in xx]
        return self.word2id[xx]

    def id22word(self, xx):
        if isinstance(xx, list):
            return [self.id2word[idx] for idx in xx]
        return self.id2word[xx]