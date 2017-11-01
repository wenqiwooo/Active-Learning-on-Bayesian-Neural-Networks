import os
import sys
import numpy as np
import tensorflow as tf
import tarfile
import urllib.request
from tqdm import tqdm
from download import maybe_download_and_extract
from cifar10 import load_training_data, load_test_data
from cifar10_bcnn import Cifar10BCNN


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('fetches', 10, 'Number of data fetches.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs for each dataset.')
flags.DEFINE_integer('classes', 10, 'Data selection size.')
flags.DEFINE_integer('batch_size', 100, 'Minibatch size.')
flags.DEFINE_integer('select_size', 10000, 'Data selection size.')


SAVE_DIR = '../checkpoint'
DATA_DIR = '../data'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def _download_if_needed(data_url, data_dir):
  maybe_download_and_extract(data_url, data_dir)


def _init_or_reset_data(data_dir):
  images, classes, _ = load_training_data()
  np.save('{}/cifar10_images'.format(data_dir), images)
  np.save('{}/cifar10_cls'.format(data_dir), classes)


def max_entropy(pred, k):
  """Returns a bitwise np array of top k highest entropy entries

  Args
    pred (np.array): [size * no of classes]
    k (int)

  Returns
    np.array: Indices of k largest.
    np.array: [size * no of classes]. 1 if entry is top k, 0 otherwise.

  """
  entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
  k_largest = np.argpartition(entropy, -k)[-k:]
  bitmap = np.zeros(entropy.shape)
  bitmap[k_largest] = 1
  return k_largest, bitmap


def random_acq(pred, k):
  pass


def max_mutual(pred, k):
  pass


def _select_data(data_dir, sess=None, model=None, f=None, initial=False):
  """Selects the next pool of data to train on
  
  Args:
    data_dir (str): Data directory
    model (Cifar10BCNN): Tensorflow model wrapper
    f (func): Acquisition function
  """
  images = np.load('{}/cifar10_images.npy'.format(data_dir))
  classes = np.load('{}/cifar10_cls.npy'.format(data_dir))
  if initial:
    selected_images = images[:FLAGS.select_size]
    selected_classes = classes[:FLAGS.select_size]
    images = images[FLAGS.select_size:]
    classes = classes[FLAGS.select_size:]
  else:
    pred = model.predict(sess, images, FLAGS.batch_size)
    indices, _ = f(pred, FLAGS.select_size)
    selected_images = images[indices]
    selected_classes = classes[indices]
    images = np.delete(images, indices, axis=0)
    classes = np.delete(classes, indices, axis=0)
  np.save('{}/cifar10_images'.format(data_dir), images)
  np.save('{}/cifar10_cls'.format(data_dir), classes)
  return selected_images, selected_classes


def main(_):
  test_images, test_classes, _ = load_test_data()
  test_images = test_images[:1000]
  test_classes = test_classes[:1000]
  images, classes = _select_data(
      DATA_DIR, initial=True)
  with tf.Session() as sess:
    model = Cifar10BCNN()
    sess.run(tf.global_variables_initializer())
    for i in range(FLAGS.fetches):
      model.optimize(sess, test_images, test_classes, FLAGS.epochs, FLAGS.batch_size)
      acc = model.validate(
          sess, test_images, test_classes, FLAGS.batch_size, FLAGS.classes, 10)
      print(acc)
      # new_images, new_classes = _select_data(DATA_DIR, sess, model, max_entropy)
      # images = np.concatenate([images, new_images], 0)
      # classes = np.concatenate([classes, new_classes], 0)
    # tf.reset_default_graph()


if __name__ == '__main__':
  # FLAGS._parse_flags()
  # _download_if_needed(DATA_URL, DATA_DIR)
  _init_or_reset_data(DATA_DIR)
  tf.app.run()








