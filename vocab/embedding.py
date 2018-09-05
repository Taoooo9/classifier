import numpy as np

def src_embedding(src_word, embedding_file):
    embedding_dim = -1
    embedding_count = 0
    with open(embedding_file, encoding = 'utf-8') as f:
        for line in f.readlines():
            if embedding_count < 1:
                data = line.split()
                embedding_dim = len(data) - 1
            embedding_count += 1

    print('\nTotal words: ' + str(embedding_count))
    print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

    find_count = 0
    embedding = np.zeros((len(src_word.word2id), embedding_dim))
    with open(embedding_file, encoding = 'utf-8') as f:
        for line in f.readlines():
            line = line.split(' ')
            if line[0] in src_word.word2id:
                vector = np.array(line[1:], dtype = 'float64')
                embedding[src_word.word2id[line[0]]] = vector
                embedding[1] += vector
                find_count += 1

    print("The number of vocab word find in extend embedding is: ", str(find_count))
    print("The number of all vocab is: ", str(len(src_word.word2id)))

    not_find = len(src_word.word2id) - find_count
    nofind_ratio = float(not_find / len(src_word.word2id))
    print('nofind_ratio: {:.4f}'.format(nofind_ratio))

    embedding[1] = embedding[1] / find_count
    embedding = embedding / np.std(embedding)

    return embedding
