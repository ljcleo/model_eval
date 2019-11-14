from time import time

import numpy as np

from .files import PRE_PROCESSED_FILE


def bayes(class_size, take_size):
    tst = time()

    preset = np.load(PRE_PROCESSED_FILE)
    vocab_size = int(np.max(preset[:, :-class_size])) + 1
    np.random.shuffle(preset)

    train_set = preset[take_size:]
    train_txt = train_set[:, :-class_size].astype(np.int64)
    train_cls = train_set[:, -class_size:].astype(np.float32)

    cls_freq = np.sum(train_cls, axis=0)
    word_freq = np.ones((vocab_size, class_size))

    for txt, cls in zip(train_txt, train_cls):
        for word in txt:
            if word == 0:
                break

            word_freq[word] += cls

    cls_freq = np.log(cls_freq / np.sum(cls_freq, axis=0))
    word_freq = np.log(word_freq / np.sum(word_freq, axis=0))

    train_time = time() - tst
    tst = time()

    test_set = preset[:take_size]
    test_txt = test_set[:, :-5].astype(np.int64)
    test_cls = test_set[:, -5:].astype(np.float32)

    res = np.array([])

    for txt, cls in zip(test_txt, test_cls):
        prob = np.copy(cls_freq)

        for word in txt:
            if word == 0:
                break

            prob += word_freq[word]

        prob = np.exp(prob - np.max(prob))
        prob /= np.sum(prob)
        res = np.append(res, cls @ prob / (np.linalg.norm(cls) * np.linalg.norm(prob)))

    test_time = time() - tst
    test_acc = np.mean(res)
    return train_time, test_time, test_acc
