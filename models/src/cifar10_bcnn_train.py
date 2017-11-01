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
flags.DEFINE_integer('fetches', 50, 'Number of data fetches.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs for each dataset.')
flags.DEFINE_integer('batch_size', 60, 'Minibatch size.')
flags.DEFINE_integer('select_size', 512, 'Data selection size.')


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


def _select_data(data_dir, sess, model, f, initial=False):
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
    pred = model.predict(sess, images, 512)
    indices, _ = f(np.concatenate(pred, 0), FLAGS.select_size)
    selected_images = images[indices]
    selected_classes = classes[indices]
    images = np.delete(images, indices)
    classes = np.delete(classes, indices)
  np.save('{}/cifar10_images'.format(data_dir), images)
  np.save('{}/cifar10_cls'.format(data_dir), classes)
  return selected_images, selected_classes


def main(_):
  model = Cifar10BCNN()
  saver = tf.train.Saver()
  with tf.Session() as sess:
    save_path = os.path.join(SAVE_DIR, 'cifar10_bcnn.ckpt')
    if os.path.exists(save_path):
      saver.restore(sess, save_path)
      initial = False
    else:
      sess.run(tf.global_variables_initializer())
      initial = True
    images, classes = _select_data(
        DATA_DIR, sess, model, max_entropy, initial=initial)
    for i in range(FLAGS.fetches):
      model.optimize(sess, images, classes, FLAGS.epochs, FLAGS.batch_size)
      saver.save(sess, save_path)
      images, classes = _select_data(DATA_DIR, sess, model, max_entropy)


if __name__ == '__main__':
  # FLAGS._parse_flags()
  # _download_if_needed(DATA_URL, DATA_DIR)
  _init_or_reset_data(DATA_DIR)
  tf.app.run()








