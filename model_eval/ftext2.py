from time import time

from fasttext import train_supervised
import numpy as np

from .files import FAST_TEXT_FILE, PRE_PROCESSED_FILE


def fast_text_2(class_size, take_size):
    tmp_file = FAST_TEXT_FILE
    label_prefix = 'C'
    use_major = True

    n_grams = 2
    init_learning_rate = 1
    train_epoch = 20

    tst = time()

    preset = np.load(PRE_PROCESSED_FILE)
    np.random.shuffle(preset)

    train_set = preset[take_size:]
    train_txt = train_set[:, :-class_size].astype(np.int64)
    train_cls = train_set[:, -class_size:].astype(np.float32)

    with open(tmp_file, 'w', encoding='utf-8') as f:
        for txt, cls in zip(train_txt, train_cls):
            for k in np.arange(class_size):
                if cls[k] != 0:
                    cls[k] = 2520 * cls[k]

            cls = cls.astype(np.int64)
            divisor = 2520

            for v in cls:
                if v > 0:
                    divisor = np.gcd(divisor, v)

            cls = cls // divisor
            labels = np.array([], dtype=np.int64)

            for k in np.arange(class_size):
                if use_major:
                    if cls[k] == np.max(cls):
                        labels = np.append(labels, k)
                else:
                    labels = np.append(labels, np.repeat(k, cls[k]))

            s = ' '.join([label_prefix + str(label) for label in labels] +
                         [str(word) for word in txt[np.nonzero(txt)[0]]])
            f.write(s + '\n')

    model = train_supervised(tmp_file, label=label_prefix, wordNgrams=n_grams, lr=init_learning_rate, epoch=train_epoch,
                             verbose=0)

    train_time = time() - tst
    tst = time()

    test_set = preset[:take_size]
    test_txt = test_set[:, :-5].astype(np.int64)
    test_cls = test_set[:, -5:].astype(np.float32)

    res = np.array([])

    for txt, cls in zip(test_txt, test_cls):
        prob = np.zeros(class_size)
        labels, sorted_prob = model.predict(' '.join([str(word) for word in txt[np.nonzero(txt)[0]]]), k=-1)
        prob[np.array([int(s[-1]) for s in labels])] = sorted_prob
        prob /= np.sum(prob)
        res = np.append(res, cls @ prob / (np.linalg.norm(cls) * np.linalg.norm(prob)))

    test_time = time() - tst
    test_acc = np.mean(res)
    return train_time, test_time, test_acc
