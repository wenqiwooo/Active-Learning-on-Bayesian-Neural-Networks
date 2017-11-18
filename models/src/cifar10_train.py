import os
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras.datasets import cifar10
from cifar10_bayesian import Cifar10BCNN, BayesianDropout, C10BetaDropout


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('fetches', 10, 'Number of data fetches.')
flags.DEFINE_integer('epochs', 40, 'Number of epochs for each dataset.')
flags.DEFINE_integer('classes', 10, 'Data selection size.')
flags.DEFINE_integer('batch_size', 50, 'Minibatch size.')
flags.DEFINE_integer('select_size', 10000, 'Data selection size.')


SAVE_DIR = '../checkpoint'
DATA_DIR = '../data'


def main(_):
  (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
  Y_train = np.squeeze(Y_train)
  Y_test = np.squeeze(Y_test)

  with tf.Session() as sess:
    model = C10BetaDropout(FLAGS.epochs, len(X_train), FLAGS.batch_size)
    saver = tf.train.Saver()
    try: 
      print('Restoring from checkpoint')
      saver.restore(sess, '../checkpoint/beta_dropout.ckpt')
    except Exception:
      print('Initializing new variables')
      sess.run(tf.global_variables_initializer())
    model.optimize(
        X_train, Y_train, FLAGS.epochs, FLAGS.batch_size,
        X_test, Y_test, 10, saver=saver)


if __name__ == '__main__':
  tf.app.run()








