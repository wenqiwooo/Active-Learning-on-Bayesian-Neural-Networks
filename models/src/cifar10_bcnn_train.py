import os
import sys
import numpy as np
import tensorflow as tf
import tarfile
import urllib.request
from download import maybe_download_and_extract
from cifar10 import load_training_data
from cifar10_bcnn import Cifar10BCNN


flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_float('max_grad_norm', 5., 'Max value of gradient.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to run.')
flags.DEFINE_integer('batch_size', 60, 'Minibatch size.')
# flags.DEFINE_integer('dim_size', 100, 'Dim of word vectors.')
# flags.DEFINE_integer('sequence_len', 120, 'Sequence length.')
# flags.DEFINE_integer('hidden_size', 100, 'Hidden layer size.')
# flags.DEFINE_integer('output_size', 9, 'Output layer size.')

DATA_DIR = '../data'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def _download_if_needed(data_url, data_dir):
  maybe_download_and_extract(DATA_URL, DATA_DIR)


def _initialize_data(data_dir):
  images, classes, _ = load_training_data()
  np.save('{}/cifar10_images'.format(data_dir), images)
  np.save('{}/cifar10_cls'.format(data_dir), classes)


def _get_data(data_dir):
  images = np.load('{}/cifar10_images.npy'.format(data_dir))
  classes = np.load('{}/cifar10_cls.npy'.format(data_dir))
  return images, classes


def main(_):
  images, classes = _get_data(DATA_DIR)
  model = Cifar10BCNN()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.optimize(sess, images, classes, FLAGS.epochs, FLAGS.batch_size)


if __name__ == '__main__':
  # _download_if_needed(DATA_URL, DATA_DIR)
  # _initialize_data(DATA_DIR)
  tf.app.run()








