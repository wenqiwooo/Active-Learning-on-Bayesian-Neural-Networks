import os
import sys
import numpy as np
import tensorflow as tf
import tarfile
import urllib.request
from tqdm import tqdm
from download import maybe_download_and_extract
from cifar10 import load_training_data, load_test_data
from keras.datasets import cifar10
from cifar10_bcnn2 import Cifar10BCNN, BayesianDropout
from cifar10_cnn import Cifar10CNN


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('fetches', 10, 'Number of data fetches.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs for each dataset.')
flags.DEFINE_integer('classes', 10, 'Data selection size.')
flags.DEFINE_integer('batch_size', 50, 'Minibatch size.')
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
  (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
  Y_train = np.squeeze(Y_train)
  Y_test = np.squeeze(Y_test)

  # saver = tf.train.Saver()
  with tf.Session() as sess:
    # model = BayesianDropout(FLAGS.epochs, len(X_train), FLAGS.batch_size)
    # sess.run(tf.global_variables_initializer())
    # model.optimize(
    #     X_train, Y_train, FLAGS.epochs, FLAGS.batch_size,
    #     X_test, Y_test, 10, sess=sess)
    # acc = model.validate(X_test, Y_test, FLAGS.batch_size, 5)
    # print(acc)

    model = Cifar10CNN()
    sess.run(tf.global_variables_initializer())
    model.optimize(sess, X_train, Y_train, FLAGS.epochs, FLAGS.batch_size)
    acc = model.validate(sess, X_test, Y_test, FLAGS.batch_size)
    print('Test accuracy: {} %'.format(acc * 100))


if __name__ == '__main__':
  # FLAGS._parse_flags()
  # _download_if_needed(DATA_URL, DATA_DIR)
  # _init_or_reset_data(DATA_DIR)
  tf.app.run()








