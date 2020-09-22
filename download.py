from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import fnmatch
import os
import sys
import tarfile
import tempfile
import unicodedata

import pandas

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow as tf
from absl import app as absl_app
from absl import flags as absl_flags
from six.moves import urllib
from sox import Transformer

URLS = {"train-clean-100": "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "train-clean-360": "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "train-other-500": "http://www.openslr.org/resources/12/train-other-500.tar.gz",
        "dev-clean": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "dev-other": "http://www.openslr.org/resources/12/dev-other.tar.gz",
        "test-clean": "http://www.openslr.org/resources/12/test-clean.tar.gz",
        "test-other": "http://www.openslr.org/resources/12/test-other.tar.gz"}


def download_and_extract(directory, url):
    if not tf.io.gfile.exists(directory):
        tf.io.gfile.makedirs(directory)

    _, tar_filepath = tempfile.mkstemp(suffix=".tar.gz")

    try:
        print(f"Downloading {url} to {tar_filepath}")

        def _progress(count, block_size, total_size):
            sys.stdout.write("\r>> Downloading {} {:.1f}%".format(
                tar_filepath, 100.0 * count * block_size / total_size))
            sys.stdout.flush()

        urllib.request.urlretrieve(url, tar_filepath, _progress)
        print()
        state_info = os.stat(tar_filepath)
        print(f"Successfully downloaded {url}, size(bytes): {state_info.st_size}")
        with tarfile.open(tar_filepath, "r") as tar:
            tar.extractall(directory)
    finally:
        tf.io.gfile.remove(tar_filepath)


def convert_audio_and_split_transcript(input_dir, source_name, target_name, output_dir, output_file):
    print(f"Pre-processing audio and transcript for {source_name}")
    source_dir = os.path.join(input_dir, source_name)
    target_dir = os.path.join(input_dir, target_name)

    if not tf.io.gfile.exists(target_dir):
        tf.io.gfile.makedirs(target_dir)

    files = []
    tfm = Transformer()
    for root, _, filenames in tf.io.gfile.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.trans.txt"):
            trans_file = os.path.join(root, filename)
            with codecs.open(trans_file, "r", "utf-8") as fin:
                for line in fin:
                    seq_id, transcript = line.split(" ", 1)
                    transcript = unicodedata.normalize("NFKD", transcript).encode(
                        "ascii", "ignore").decode("ascii", "ignore").strip().lower()

                    flac_file = os.path.join(root, seq_id + ".flac")
                    wav_file = os.path.join(target_dir, seq_id + ".wav")
                    if not tf.io.gfile.exists(wav_file):
                        tfm.build(flac_file, wav_file)
                    wav_filesize = os.path.getsize(wav_file)

                    files.append((os.path.abspath(wav_file), wav_filesize, transcript))

    csv_file_path = os.path.join(output_dir, output_file)
    df = pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])
    df.to_csv(csv_file_path, index=False, sep="\t")
    print(f"Successfully generated csv file {csv_file_path}")


def download_and_process_datasets(directory, datasets):
    print("Preparing LibriSpeech dataset: {}".format(",".join(datasets)))
    for dataset in datasets:
        print(f"Preparing dataset {dataset}")
        dataset_dir = os.path.join(directory, dataset)
        download_and_extract(dataset_dir, URLS[dataset])
        convert_audio_and_split_transcript(dataset_dir + "/LibriSpeech", dataset, dataset + "-wav",
                                           dataset_dir + "/LibriSpeech", dataset + ".csv")


def define_data_download_flags():
    absl_flags.DEFINE_string("data_dir", "../Dataset", "Directory to download data and extract the tarball")
    absl_flags.DEFINE_bool("train_only", False, "If true, only download the training set")
    absl_flags.DEFINE_bool("dev_only", False, "If true, only download the dev set")
    absl_flags.DEFINE_bool("test_only", True, "If true, only download the test set")


def main(_):
    if not tf.io.gfile.exists(FLAGS.data_dir):
        tf.io.gfile.makedirs(FLAGS.data_dir)

    if FLAGS.train_only:
        download_and_process_datasets(FLAGS.data_dir, ["train-clean-100", "train-clean-360", "train-other-500"])
    elif FLAGS.dev_only:
        download_and_process_datasets(FLAGS.data_dir, ["dev-clean", "dev-other"])
    elif FLAGS.test_only:
        download_and_process_datasets(FLAGS.data_dir, ["test-clean", "test-other"])
    else:
        # By default we download the entire dataset.
        download_and_process_datasets(FLAGS.data_dir, URLS.keys())


if __name__ == "__main__":
    define_data_download_flags()
    FLAGS = absl_flags.FLAGS
    absl_app.run(main)
