from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend


def build_model(input_dim, output_dim, rnn_units=800, random_state=1) -> models.Model:
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # Define input tensor [batch, time, frequency, channel]
    inputs = layers.Input([None, input_dim, 1], name='inputs')

    x = layers.Conv2D(32, [11, 41], [2, 2], 'same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='activation1')(x)

    x = layers.Conv2D(32, [11, 21], [1, 2], 'same', use_bias=False, name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='activation2')(x)

    # We need to squeeze to 3D tensor. Thanks to the stride in frequency domain,
    # we reduce the number of features four times for each channel.
    shape = backend.int_shape(x)
    x = layers.Reshape([-1, shape[-2] * shape[-1]])(x)

    for i in [1, 2, 3, 4, 5]:
        rnn = layers.GRU(rnn_units, return_sequences=True, name=f'gru{i}')
        x = layers.Bidirectional(rnn, name=f'rnn{i}')(x)

    x = layers.TimeDistributed(layers.Dense(units=rnn_units * 2), name='fc')(x)
    x = layers.ReLU(name='activation3')(x)
    x = layers.Dropout(rate=0.5, name='dropout')(x)
    outputs = layers.TimeDistributed(layers.Dense(output_dim), name='outputs')(x)

    return models.Model(inputs, outputs, name='Model')


def get_loss() -> Callable:
    def get_length(tensor):
        lengths = tf.math.reduce_sum(tf.ones_like(tensor), 1)
        return tf.cast(lengths, tf.int32)

    def ctc_loss(labels, logits):
        label_length = get_length(labels)
        logit_length = get_length(tf.math.reduce_max(logits, 2))
        return tf.nn.ctc_loss(labels, logits, label_length, logit_length, False, blank_index=-1)

    return ctc_loss
