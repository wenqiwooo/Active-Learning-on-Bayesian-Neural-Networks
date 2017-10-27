import numpy as np
import tensorflow as tf
from model import NerModel


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('max_grad_norm', 5., 'Max value of gradient.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to run.')
flags.DEFINE_integer('batch_size', 60, 'Minibatch size.')
flags.DEFINE_integer('dim_size', 100, 'Dim of word vectors.')
flags.DEFINE_integer('sequence_len', 120, 'Sequence length.')
flags.DEFINE_integer('hidden_size', 100, 'Hidden layer size.')
flags.DEFINE_integer('output_size', 9, 'Output layer size.')

DATA_DIR = '../data'


def _get_data(data_dir):
  X = np.load('{}/input.train.npy'.format(data_dir))
  Y = np.load('{}/output.train.npy'.format(data_dir))
  return X, Y


def _get_glove(data_dir):
  glove = np.load('{}/glove.npy'.format(data_dir))
  return glove


def main(_):
  X, Y = _get_data(DATA_DIR)
  glove = _get_glove(DATA_DIR)
  model = NerModel(
      embeddings=glove,
      dim_size=FLAGS.dim_size,
      seq_len=FLAGS.sequence_len,
      hidden_size=FLAGS.hidden_size,
      output_size=FLAGS.output_size,
      lr=FLAGS.learning_rate,
      max_grad_norm=FLAGS.max_grad_norm)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.optimize(sess, X, Y, FLAGS.epochs, FLAGS.batch_size)


class Trainer(object):
  def __init__(self):
    self.X, self.Y = _get_data(DATA_DIR)
    self.glove = _get_glove(DATA_DIR)

  def train(self, hidden_size, lr, batch_size):
    FLAGS._parse_flags()
    with tf.Graph().as_default(), tf.Session() as sess:
      model = NerModel(
        embeddings=self.glove,
        dim_size=FLAGS.dim_size,
        seq_len=FLAGS.sequence_len,
        output_size=FLAGS.output_size,
        max_grad_norm=FLAGS.max_grad_norm,
        hidden_size=hidden_size,
        lr=lr)
      sess.run(tf.global_variables_initializer())
      loss = model.optimize(sess, self.X, self.Y, FLAGS.epochs, batch_size)
      return loss


if __name__ == '__main__':
  tf.app.run()








