from time import time
from typing import List, Tuple

import numpy as np

from .files import PRE_PROCESSED_FILE


def knn(class_size, take_size):
    k = 20

    tst = time()

    preset = np.load(PRE_PROCESSED_FILE)
    vocab_size = int(np.max(preset[:, :-class_size])) + 1
    np.random.shuffle(preset)

    train_set = preset[take_size:]
    train_txt = train_set[:, :-class_size].astype(np.int64)
    train_cls = train_set[:, -class_size:].astype(np.float32)

    doc_size = train_set.shape[0]
    word_link: List[List[Tuple[int, float]]] = [[] for _ in np.arange(vocab_size)]
    train_vec_norm = np.zeros(doc_size)
    df = np.ones(vocab_size).astype(np.float32)

    for i in np.arange(doc_size):
        d = {}
        txt_size = 0

        for word in train_txt[i]:
            if word == 0:
                break

            d[word] = d.get(word, 0) + 1
            txt_size += 1

        for word in d:
            word_link[word].append((i, d[word] / txt_size))

        for word in set(train_txt[i]).difference({0}):
            df[word] += 1

    idf = np.log(doc_size / df)

    for word in np.arange(1, vocab_size):
        for k in np.arange(len(word_link[word])):
            i, v = word_link[word][k]
            v *= idf[word]

            word_link[word][k] = i, v
            train_vec_norm[i] += v ** 2

    train_vec_norm = np.sqrt(train_vec_norm)

    train_time = time() - tst
    tst = time()

    test_set = preset[:take_size]
    test_txt = test_set[:, :-5].astype(np.int64)
    test_cls = test_set[:, -5:].astype(np.float32)

    res = np.array([])

    for txt, cls in zip(test_txt, test_cls):
        sim = np.zeros(doc_size)
        prob = np.zeros(class_size)

        d = {}
        txt_size = 0
        norm = 0

        for word in txt:
            if word == 0:
                break

            d[word] = d.get(word, 0) + 1
            txt_size += 1

        for word in d:
            x = d[word] / txt_size * idf[word]
            norm += x ** 2

            for i, v in word_link[word]:
                sim[i] += v * x

        for i in np.arange(doc_size):
            sim[i] /= norm * train_vec_norm[i]

        for i in np.argsort(sim)[-k:]:
            prob += train_cls[i]

        prob /= np.sum(prob)
        res = np.append(res, cls @ prob / (np.linalg.norm(cls) * np.linalg.norm(prob)))

    test_time = time() - tst
    test_acc = np.mean(res)
    return train_time, test_time, test_acc
