from collections import Counter
from random import random
from time import time

import numpy as np

from .files import PRE_PROCESSED_FILE, get_feature_file


def feature(class_size):
    t = 5e-5
    feature_scale = {'small': 3000, 'large': 10000}

    tst = time()

    preset = np.load(PRE_PROCESSED_FILE)
    txt = preset[:, :-class_size].astype(np.int64)
    cls = preset[:, -class_size:].astype(np.float32)
    pad_size = txt.shape[1]
    vocab_set = Counter()

    for s in txt:
        vocab_set.update([s[0]])

        for i in np.arange(1, pad_size):
            if s[i] == 0:
                break

            vocab_set.update([s[i], (s[i - 1], s[i])])

    total = sum(vocab_set.values())

    for gram in vocab_set:
        d = t * total / vocab_set[gram]

        if random() > np.sqrt(d) + d:
            vocab_set[gram] = 0

    for scale in feature_scale:
        good_dict = {gram[0]: i + 1 for i, gram in enumerate(vocab_set.most_common(feature_scale[scale]))}
        pairs = []
        pad_len = 0

        for t in txt:
            bag = set()

            if t[0] in good_dict:
                bag.add(good_dict[t[0]])

            for i in np.arange(1, pad_size):
                if t[i] == 0:
                    break

                if t[i] in good_dict:
                    bag.add(good_dict[t[i]])

                if (t[i - 1], t[i]) in good_dict:
                    bag.add(good_dict[(t[i - 1], t[i])])

            if len(bag) == 0:
                continue

            bag = np.array(list(bag))
            pad_len = max(pad_len, len(bag))
            pairs.append(bag)

        for i in np.arange(len(pairs)):
            pairs[i] = np.append(np.pad(pairs[i], [0, pad_len - len(pairs[i])], 'constant', constant_values=0), cls[i])

        np.save(get_feature_file(scale), np.array(pairs))

    feature_time = time() - tst
    return feature_time
