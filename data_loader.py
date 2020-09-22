from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs

import numpy as np
import soundfile
import tensorflow as tf
import multiprocessing
import config


def compute_spectrogram_feature(samples, max_freq=None, eps=1e-14):
    if max_freq is None:
        max_freq = config.sample_rate / 2
    if max_freq > config.sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of sample rate.")

    if config.stride_ms > config.window_ms:
        raise ValueError("Stride size must not be greater than window size.")

    stride_size = int(0.001 * config.sample_rate * config.stride_ms)
    window_size = int(0.001 * config.sample_rate * config.window_ms)

    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    n_shape = (window_size, (len(samples) - window_size) // stride_size + 1)
    n_strides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape=n_shape, strides=n_strides)
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2
    scale = np.sum(weighting ** 2) * config.sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale

    frequency = float(config.sample_rate) / window_size * np.arange(fft.shape[0])

    ind = np.where(frequency <= max_freq)[0][-1] + 1
    spectrogram = np.log(fft[:ind, :] + eps)
    return np.transpose(spectrogram, (1, 0))


def compute_label_feature(text, token_to_idx):
    tokens = list(text.strip().lower())
    feats = [token_to_idx[token] for token in tokens]
    return feats


class TextFeaturizer(object):

    def __init__(self, vocab_file):
        super(TextFeaturizer, self).__init__()
        lines = []
        with codecs.open(vocab_file, "r", "utf-8") as fin:
            lines.extend(fin.readlines())
        self.token_to_index = {}
        self.index_to_token = {}
        self.speech_labels = ""
        index = 0
        for line in lines:
            line = line[:-1]
            if line.startswith("#"):
                continue
            self.token_to_index[line] = index
            self.index_to_token[index] = line
            self.speech_labels += line
            index += 1


def normalize_audio_feature(audio_feature):
    mean = np.mean(audio_feature, axis=0)
    var = np.var(audio_feature, axis=0)
    normalized = (audio_feature - mean) / (np.sqrt(var) + 1e-6)
    return normalized


def preprocess_audio(audio_file_path):
    data, _ = soundfile.read(audio_file_path)
    data = (data * np.iinfo(np.int16).max).astype(np.int16)
    feature = compute_spectrogram_feature(data)
    if config.normalize:
        feature = normalize_audio_feature(feature)
    feature = np.expand_dims(feature, axis=2)
    return feature


def preprocess_data(file_path):
    with tf.io.gfile.GFile(file_path, "r") as f:
        lines = f.read().splitlines()
    lines = lines[1:]
    lines = [line.split("\t", 2) for line in lines]
    lines.sort(key=lambda item: int(item[1]))
    return [tuple(line) for line in lines]


def input_fn():
    data_entries = preprocess_data(config.train_path)
    text_featurizer = TextFeaturizer(config.vocabulary_path)

    def generator():
        for path, _, transcript in data_entries:
            audio = preprocess_audio(path)
            label = compute_label_feature(transcript, text_featurizer.token_to_index)
            yield audio, label

    dataset = tf.data.Dataset.from_generator(generator,
                                             (tf.float32, tf.int32),
                                             (tf.TensorShape([None, config.input_dim, 1]), tf.TensorShape([None])))
    dataset = dataset.repeat(config.nb_epochs)

    dataset = dataset.padded_batch(config.batch_size,
                                   (tf.TensorShape([None, config.input_dim, 1]), tf.TensorShape([None])))

    dataset = dataset.prefetch(40 * config.batch_size)

    return dataset
