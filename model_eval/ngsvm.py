from time import time
from typing import List, Set

import numpy as np
from svm import svm_problem, svm_parameter
from svmutil import svm_train, svm_predict

from .files import get_feature_file


def n_gram_svm(class_size, take_size):
    cost = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    gamma = np.array([0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125])

    tst = time()

    preset = np.load(get_feature_file('small'))
    vocab_size = int(np.max(preset[:, :-class_size])) + 1
    np.random.shuffle(preset)

    train_set = preset[take_size:]
    train_txt = train_set[:, :-class_size].astype(np.int64)
    train_cls = train_set[:, -class_size:].astype(np.float32)

    train_dict = [{gram: 1 if gram in txt else 0 for gram in np.arange(1, vocab_size)} for txt in train_txt]
    train_major: List[Set[int]] = [set() for _ in np.arange(class_size)]

    for i, cls in enumerate(train_cls):
        for k in np.nonzero(np.abs(cls - np.max(cls)) < 1e-4)[0]:
            train_major[k].add(i)

    models = []

    for k in np.arange(class_size):
        problem = svm_problem([1 if i in train_major[k] else -1 for i in np.arange(len(train_cls))], train_dict)
        param = svm_parameter('-t 0 -c %f -g %f -b 1 -q' % (cost[k], gamma[k]))
        models.append(svm_train(problem, param))

    train_time = time() - tst
    tst = time()

    test_set = preset[:take_size]
    test_txt = test_set[:, :-5].astype(np.int64)
    test_cls = test_set[:, -5:].astype(np.float32)

    res = np.array([])
    test_dict = [{gram: 1 if gram in txt else 0 for gram in np.arange(1, vocab_size)} for txt in test_txt]

    for dic, cls in zip(test_dict, test_cls):
        prob = np.zeros(class_size)

        for k in np.arange(class_size):
            _, _, p = svm_predict([], [dic], models[k], '-b 1 -q')
            prob[k] = p[0][0]

        prob /= np.sum(prob)
        res = np.append(res, cls @ prob / (np.linalg.norm(cls) * np.linalg.norm(prob)))

    test_time = time() - tst
    test_acc = np.mean(res)
    return train_time, test_time, test_acc
