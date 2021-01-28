import numpy as np
from random import choices, randint

def count_vectorizer(lines: list):
    documents = [line.lower().split() for line in lines]
    vocabulary = set()
    for document in documents:
        vocabulary |= set(document)
    vocabulary = sorted(vocabulary)
    word2index = {vocabulary[i]: i for i in range(len(vocabulary))}
    X = np.zeros(shape=(len(documents), len(vocabulary)), dtype=int)
    for i in range(len(documents)):
        for word in documents[i]:
            X[i, word2index[word]] += 1
    return X, vocabulary

def latent_dirichlet_allocation(X, k: int=20,
        alpha: float=0.02, beta: float=0.1, max_iter: int=500):
    if not ((isinstance(X, np.matrix) or isinstance(X, np.ndarray))
            and isinstance(k, int)):
        raise TypeError('X must be matrix or ndarray and k must int')
    if X.shape[0] == 0 or X.shape[1] == 0 or k <= 0:
        raise ValueError('all of X.shape[0], X.shape[1] and k must be positive')

    w = X.shape[1]
    doc, word, topic, n_d_k, n_w_k, n_k = random_init(X, k)

    for curr in range(max_iter):
        print(F"Iteration: {curr}/{max_iter}")
        for i in range(len(topic)):
            n_d_k[doc[i], topic[i]] -= 1
            n_w_k[word[i], topic[i]] -= 1
            n_k[topic[i]] -= 1
            p = [
                (n_d_k[doc[i], k_] + alpha) * \
                (n_w_k[word[i], k_] + beta) / \
                (n_k[k_] * beta * w)
                for k_ in range(k)]
            topic[i] = choices(range(k), p)
            n_d_k[doc[i], topic[i]] += 1
            n_w_k[word[i], topic[i]] += 1
            n_k[topic[i]] += 1
    return topic, n_d_k, n_w_k, n_k

def random_init(X, k: int):
    doc, word, topic = [], [], []
    for row, col in zip(*X.nonzero()):
        for i in range(X[row, col]):
            doc.append(row)
            word.append(col)
            topic.append(randint(0, k - 1))

    n_d_k = np.zeros(shape=(X.shape[0], k), dtype=int)
    n_w_k = np.zeros(shape=(X.shape[1], k), dtype=int)
    n_k = np.zeros(shape=(k), dtype=int)

    for i in range(len(topic)):
        n_d_k[doc[i], topic[i]] += 1
        n_w_k[word[i], topic[i]] += 1
        n_k[topic[i]] += 1

    return doc, word, topic, n_d_k, n_w_k, n_k

with open('test.txt', 'rt') as file:
    lines = file.readlines()
    assert int(lines[0]) + 1 == len(lines)
    lines = lines[1:]

X, vocabulary = count_vectorizer(lines)
k = 3
topic, n_d_k, n_w_k, n_k = latent_dirichlet_allocation(X, k=k)
for j in range(k):
    print(F'in topic {j}:')
    for i in n_w_k[:, j].argsort()[-10:]:
        print(F'\t{vocabulary[i]}: {n_w_k[i, j]}')
