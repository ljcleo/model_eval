from argparse import ArgumentParser, REMAINDER
from csv import writer
from logging import Formatter, FileHandler, getLogger, StreamHandler
from sys import stdout

import numpy as np
from tensorflow import config as cfg

from .bayes import bayes
from .cnn import cnn
from .feature import feature
from .ftext import fast_text
from .ftext2 import fast_text_2
from .files import init, clean
from .knn import knn
from .lstm import lstm
from .ngsvm import n_gram_svm
from .pre_process import pre_process

INDICES = ['train_time', 'test_time', 'accuracy']
MODEL_CONFIG = {
    'bayes': (bayes, False, 'naive Bayes'),
    'knn': (knn, False, 'k-nearest neighbor'),
    'svm': (n_gram_svm, True, 'n-gram SVM'),
    'lstm': (lstm, False, 'LSTM'),
    'ft': (fast_text, True, 'FastText'),
    'ft2': (fast_text_2, False, 'FastText (CPU)'),
    'cnn': (cnn, False, 'CNN')
}

CLASS_SIZE = 5

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-w', '--word-trunc', type=int, default=500)
    parser.add_argument('-f', '--freq-min', type=int, default=1)
    parser.add_argument('-s', '--skip-word', action='store_true')
    parser.add_argument('-t', '--take-size', type=int, default=150)
    parser.add_argument('-o', '--output-file', type=str, default='eval.csv')
    parser.add_argument('-l', '--log-file', type=str, default='eval.log')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('round', type=int)
    parser.add_argument('models', type=str, nargs=REMAINDER, choices=MODEL_CONFIG.keys())

    args = parser.parse_args()
    word_trunc = args.word_trunc
    min_freq = args.freq_min
    use_skip = args.skip_word
    take_size = args.take_size
    output_file = args.output_file
    log_file = args.log_file
    is_debug = args.debug
    eval_round = args.round
    models = args.models

    if len(models) == 0:
        models = MODEL_CONFIG.keys()

    for gpu in cfg.experimental.list_physical_devices('GPU'):
        cfg.experimental.set_memory_growth(gpu, True)

    logger = getLogger('Logger')
    logger.setLevel('INFO')

    formatter = Formatter('[%(asctime)s] %(levelname)s: %(message)s')

    file_handler = FileHandler(log_file, 'w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if is_debug:
        stream_handler = StreamHandler(stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info(f'Total evaluation round: {eval_round}')
    logger.info(f'Models to be evaluated: {" ".join(models)}')
    logger.info(f'Parameters: word-trunc {word_trunc} freq-min {min_freq} skip-word {use_skip} take-size {take_size}')
    logger.info('Start evaluating ...')

    eval_stat = {model: {index: np.array([]) for index in INDICES} for model in models}
    average_stat = {model: {} for model in models}

    init()
    pre_process(word_trunc, min_freq, use_skip)
    logger.info('Pre-processing completed.')

    for t in np.arange(eval_round):
        logger.info(f'Round {t + 1}:')

        for model in models:
            if MODEL_CONFIG[model][1]:
                feature_time = feature(CLASS_SIZE)
                logger.info(f'Feature training time: {feature_time:.4f}s')
                break
        else:
            feature_time = 0
            logger.info('No models require feature training. Skipped.')

        for model in models:
            logger.info(f'Model: {MODEL_CONFIG[model][2]}')
            train_time, test_time, accuracy = MODEL_CONFIG[model][0](CLASS_SIZE, take_size)

            if MODEL_CONFIG[model][1]:
                train_time += feature_time

            eval_stat[model]['train_time'] = np.append(eval_stat[model]['train_time'], train_time)
            eval_stat[model]['test_time'] = np.append(eval_stat[model]['test_time'], test_time)
            eval_stat[model]['accuracy'] = np.append(eval_stat[model]['accuracy'], accuracy)

            logger.info(f'Training time: {train_time:.4f}s')
            logger.info(f'Testing time: {test_time:.4f}s')
            logger.info(f'Accuracy: {accuracy:.4%}')

    for model in models:
        for index in INDICES:
            average_stat[model][index] = np.mean(eval_stat[model][index])

    logger.info('Finish evaluating. Writing CSV file ...')

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        output = writer(f)
        output.writerow(['model'] + INDICES)

        for model in models:
            output.writerow([MODEL_CONFIG[model][2]] + [average_stat[model][index] for index in INDICES])

    clean()
    logger.info('Done.')
