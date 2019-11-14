from time import time

import numpy as np
import tensorflow as tf

from .files import get_feature_file


def fast_text(class_size, take_size):
    buffer_size = 50000
    batch_size = [50, 10, 1]

    embed_size = 10
    dropout_rate = 0.3

    init_learning_rate = 0.001
    decay_step = 500
    decay_rate = 0.96

    train_epoch = 100
    min_delta = 1e-2

    tst = time()

    preset = np.load(get_feature_file('large'))
    vocab_size = int(np.max(preset[:, :-class_size]) + 1)
    data_set = tf.data.Dataset.from_tensor_slices((tf.cast(preset[:, :-class_size], tf.int64),
                                                  tf.cast(preset[:, -class_size:], tf.float32)))
    data_set = data_set.shuffle(buffer_size, reshuffle_each_iteration=False)

    train_data = data_set.skip(take_size * 2).shuffle(buffer_size).batch(batch_size[0]).repeat()
    valid_data = data_set.skip(take_size).take(take_size).batch(batch_size[1]).repeat()
    test_data = data_set.take(take_size).batch(batch_size[2]).repeat()

    inputs = tf.keras.Input(shape=(preset.shape[1] - 5,))
    embed = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)(inputs)
    dropout = tf.keras.layers.Dropout(dropout_rate)(embed)
    mean = tf.reduce_mean(dropout, axis=1)
    outputs = tf.keras.layers.Dense(class_size, activation='softmax')(mean)

    decay = tf.keras.optimizers.schedules.ExponentialDecay(init_learning_rate, decay_step, decay_rate, staircase=True)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=decay),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CosineSimilarity()])

    with tf.device('/CPU:0'):
        model.fit(train_data, epochs=train_epoch, steps_per_epoch=(preset.shape[0] - take_size * 2) // batch_size[0],
                  validation_data=valid_data, validation_steps=take_size // batch_size[1], verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=5,
                                                              verbose=0)])

    train_time = time() - tst
    tst = time()

    _, test_acc = model.evaluate(test_data, steps=take_size // batch_size[2], verbose=0)
    test_time = time() - tst
    return train_time, test_time, test_acc
