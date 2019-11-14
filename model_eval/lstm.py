from time import time

import numpy as np
import tensorflow as tf

from .files import PRE_PROCESSED_FILE


def lstm(class_size, take_size):
    buffer_size = 50000
    batch_size = [50, 10, 1]

    use_gru = False
    use_bi_rnn = True

    embed_size = 10
    hidden_size = [16]
    dense_size = [32]
    dropout_rate = 0.3

    init_learning_rate = 0.001
    decay_step = 100
    decay_rate = 0.8

    train_epoch = 50
    min_delta = 1e-2

    tst = time()

    preset = np.load(PRE_PROCESSED_FILE)
    vocab_size = int(np.max(preset[:, :-class_size]) + 1)
    data_set = tf.data.Dataset.from_tensor_slices((tf.cast(preset[:, :-class_size], tf.int64),
                                                  tf.cast(preset[:, -class_size:], tf.float32)))
    data_set = data_set.shuffle(buffer_size, reshuffle_each_iteration=False)

    train_data = data_set.skip(take_size * 2).shuffle(buffer_size).batch(batch_size[0]).repeat()
    valid_data = data_set.skip(take_size).take(take_size).batch(batch_size[1]).repeat()
    test_data = data_set.take(take_size).batch(batch_size[2]).repeat()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True))

    for i in np.arange(len(hidden_size)):
        cell_type = tf.keras.layers.GRU if use_gru else tf.keras.layers.LSTM
        layer = cell_type(hidden_size[i], return_sequences=i < len(hidden_size) - 1)

        if use_bi_rnn:
            layer = tf.keras.layers.Bidirectional(layer)

        model.add(layer)

    for i in np.arange(len(dense_size)):
        model.add(tf.keras.layers.Dense(dense_size[i], activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(class_size, activation='softmax'))

    decay = tf.keras.optimizers.schedules.ExponentialDecay(init_learning_rate, decay_step, decay_rate, staircase=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=decay),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CosineSimilarity()])
    model.fit(train_data, epochs=train_epoch, steps_per_epoch=(preset.shape[0] - take_size * 2) // batch_size[0],
              validation_data=valid_data, validation_steps=take_size // batch_size[1], verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=2,
                                                          verbose=0)])

    train_time = time() - tst
    tst = time()

    _, test_acc = model.evaluate(test_data, steps=take_size // batch_size[2], verbose=0)
    test_time = time() - tst
    return train_time, test_time, test_acc
