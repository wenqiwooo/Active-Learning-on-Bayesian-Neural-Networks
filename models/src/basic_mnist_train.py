import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from basic_mnist import MnistModel


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to run.')
flags.DEFINE_integer('batch_size', 100, 'Minibatch size.')
flags.DEFINE_integer('input_size', 784, 'Dim of word vectors.')
flags.DEFINE_integer('output_size', 10, 'Output layer size.')
flags.DEFINE_string('data_dir', '../data', 'Data directory.')


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  model = MnistModel(
      input_dim=FLAGS.input_size,
      output_dim=FLAGS.output_size,
      lr=FLAGS.learning_rate)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.optimize(sess, mnist, FLAGS.epochs, FLAGS.batch_size)


if __name__ == '__main__':
  tf.app.run()








