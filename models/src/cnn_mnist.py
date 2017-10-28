from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

class MnistCNNModel(object):
    """
    Adapted from Tensorflow tutorial on building Layers.
    """

    def __init__(self, input_dim, output_dim, lr=None):
        """

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            lr (float): learning rate
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._lr = lr
        self._layers = None
        self._clf = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir="/tmp/mnist_convnet_model",
        )

    def optimize(self, mnist, epochs=None, batch_size=100):
        """
        Fits the data to CNN.

        Args:
            mnist (tf.dataset): Tensorflow MNIST dataset
            epochs (int): epochs to train CNN for
            batch_size(int): batch size :)
        """
        # Load training and eval data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=batch_size,
            num_epochs=epochs,
            shuffle=True
        )

        self._clf.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook],
        )

    def evaluate(self, mnist):
        """
        Evaluate our MNIST on the trained CNN.

        Args:
            mnist (tf.dataset): Tensorflow MNIST dataset
        """
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        # Evaluate the model and return results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False,
        )

        eval_results = self._clf.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    def _model_fn(self, features, labels, mode):
        """
        The model function to be passed into Tensorflow Estimators
        """

        # Build our graph if it has yet to be built
        if self._layers == None:
            self._build(
                features,
                tf.estimator.ModeKeys.TRAIN,
            )

        return self._get_spec(labels, mode)

    def _build(self, features, mode):
        """
        Builds the Tensorflow layers for the neural network.
        """

        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
        )

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
        )

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )

        self._layers = tf.layers.dense(inputs=dropout, units=10)

    def _get_spec(self, labels, mode):
        """
        Builds the estimator spec for the optimization/prediction.

        Returns (tf.estimator.EstimatorSpec): the specifications for a TF
            Estimator.
        """

        # TF graph must be built before this method is called.
        assert self._layers != None

        predictions = {
            "classes": tf.argmax(input=self._layers, axis=1),
            "probabilities": tf.nn.softmax(self._layers, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
            )

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(
            indices=tf.cast(labels, tf.int32),
            depth=10,
        )

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=self._layers,
        )


        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step(),
            )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
            )

        # Add evaluation metrics (for EVAL mode)
        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"],
        )

        eval_metric_ops = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
        )

def main(_):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    cnn_mnist = MnistCNNModel(784, 10)
    # cnn_mnist.optimize(mnist)
    cnn_mnist.evaluate(mnist)


if __name__ == "__main__":
    tf.app.run()
