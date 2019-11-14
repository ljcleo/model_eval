from collections import Counter
from csv import reader
from string import punctuation as en_pun

import numpy as np
import tensorflow_datasets as tfds
from jieba import cut
from snownlp import SnowNLP
from zhon.hanzi import punctuation as zh_pun

from .files import DATA_SET_FILE, PRE_PROCESSED_FILE, SKIP_WORD_FILE, STYLE_TRANS_FILE


def pre_process(word_trunc, min_freq, use_skip):
    g1 = str.maketrans(''.join(map(chr, range(0xff01, 0xff5f))) + '\u3000', ''.join(map(chr, range(0x21, 0x7f))) + ' ')
    p = en_pun + zh_pun
    g2 = str.maketrans(p, ' ' * len(p))

    with open(STYLE_TRANS_FILE, 'r', encoding='utf-8') as f:
        style_trans = eval(f.read())

    if use_skip:
        with open(SKIP_WORD_FILE, 'r', encoding='utf-8') as f:
            k = set(f.read().strip().split())
    else:
        k = {}

    txt = []
    cls = []

    with open(DATA_SET_FILE, 'r', encoding='utf-8') as f:
        r = reader(f)
        next(r)

        for t in r:
            c = np.zeros(5)

            for x in eval(t[2]):
                if style_trans[x] != -1:
                    c[style_trans[x]] += 1

            if np.sum(c) == 0:
                continue

            cls.append(c / np.sum(c))

            s = ' '.join(cut(SnowNLP((t[0] + ' ' + t[3]).translate(g1).translate(g2).lower()).han, HMM=True)).strip()
            txt.append([w for w in s.split() if w not in k][:word_trunc])

    vocab_set = Counter()
    pad_len = 0

    for s in txt:
        vocab_set.update(s)
        pad_len = max(pad_len, len(s))

    encoder = tfds.features.text.TokenTextEncoder({word for word in vocab_set if vocab_set[word] >= min_freq})
    pairs = []

    for i in np.arange(len(txt)):
        enc = encoder.encode(' '.join(txt[i]))
        padded = np.pad(enc, [0, pad_len - len(enc)], 'constant', constant_values=0)
        pairs.append(np.append(padded, cls[i]))

    np.save(PRE_PROCESSED_FILE, np.array(pairs))
