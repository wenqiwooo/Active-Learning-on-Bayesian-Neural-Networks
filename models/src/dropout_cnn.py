import os
import sys
import datetime

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from inference_dropout import InferenceDropout as Dropout

num_classes = 10
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


class BayesianCNN(object):
    """
    Wrapper around Keras model.
    """

    def __init__(self, loss, optimizer):
        """
        Initializes the Keras model.
        """

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        self._model = model

    def optimize(self, x, y, epochs=12, batch_size=32):
        """
        Fits the model given x and y.

        Params:
            - x: input data
            - y: label data
            - epochs: times to train the neural network
            - batch_size: how big of a batch to train each time

        Returns: None
        """

        self._model.fit(x, y, epochs=epochs, batch_size=batch_size)

    def optimize_batch(self, batch_x, batch_y):
        """
        Fits the model to a single batch of x and y.

        Params:
            - batch_x: a single batch of input data
            - batch_y: a single batch of output data

        Returns: None
        """

        self._model.train_on_batch(batch_x, batch_y)

    def predict(self, x, batch_size=32):
        """
        Returns prediction given input data.

        Params:
            - x: input data
            - batch_size: how big of a batch to predict each iteration

        Returns: 2D array of predictions, one row for each row of input x
        """

        return self._model.predict(x, batch_size=batch_size)

    def predict_batch(self, x):
        """
        Returns prediction given a single batch of input data.

        Params:
            - x: input data

        Returns: 2D array of predictions, one row for each row of input x
        """

        return self._model.predict_on_batch(x)

    def evaluate(self, x, y):
        """
        Returns the loss and accuracy of the model.

        Params:
            - x: test input data
            - y: test label data

        Returns:
            - loss: loss value
            - accuracy: accuracy value
        """

        return self._model.evaluate(x, y, verbose=0)

    def sample(self, x, num_samples=10):
        """
        Gets the predictive mean and predictive variance of the neural network.
        This works because of how Dropout layers can be seen as placing a prior
        on the weights of a Neural Network.

        Params:
            - x: input data x

        Returns (2D array, 2D array):
            - predictive_mean
            - predictive_variance
        """

        probs = []
        for _ in range(num_samples):
            probs += [self._model.predict(x)]

        predictive_mean = np.mean(probs, axis=0)
        predictive_variance = np.var(probs, axis=0)

        return predictive_mean, predictive_variance

    def load_model(self, filename):
        """
        Loads a model from a saved pre-trained Keras model.

        Params:
            - filename: .h5 or equivalent file saved by Keras

        Returns: None
        """

        self._model = load_model(
            filename,
            custom_objects={'InferenceDropout': Dropout},
        )
        print(f'Keras model loaded from {filename}.')

    def save_model(self, save_dir, filename):
        """
        Saves our Keras model as a h5 file. NOTE: Requires h5py.

        Params:
            - save_dir: directory to save the model in
            - filename: name to store the model as
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, filename)
        self._model.save(model_path)
        print('Saved trained model at %s ' % model_path)


def sum_of_mean_square_errors(var):
    """
    Returns the sum of mean square errors criterion.
    TODO: Figure out the crietrion here!

    Params:
        - var: predictive variance from classifier

    Returns: criterion value
    """

    return np.sum(var)


def active_learn(model, init_x, init_y, unobserved_x, unobserved_y, iters=100):
    """
    Starts an active learning process to train the model
    """
    for i in range(iters):
        print(f'Running active learning iteration {i+1}')

        # Naive selection: just choosing ONE data point
        min_var = np.inf
        idx = None
        for _ in range(10):
            j = np.random.randint(low=0, high=len(unobserved_x))
            _, var = model.sample(np.delete(unobserved_x, j, axis=0))

            # TODO: I'm not sure if we should sum up the variance like that
            s = sum_of_mean_square_errors(var)
            if s < min_var:
                min_var = s
                idx = j

        # Add that to our training data
        init_x = np.append(init_x, np.take(unobserved_x, [idx], axis=0), axis=0)
        init_y = np.append(init_y, np.take(unobserved_y, [idx], axis=0), axis=0)

        print(f'Total data used so far: {init_x.shape[0]}')

        # Optimize the model again
        model.optimize(init_x, init_y)

        # Remove from unobserved data
        unobserved_x = np.delete(unobserved_x, idx, 0)
        unobserved_y = np.delete(unobserved_y, idx, 0)


def main():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Initialize a model
    m = BayesianCNN(keras.losses.kullback_leibler_divergence,
                    keras.optimizers.Adadelta())

    # We initially train the model with only 50 inputs
    init_x, init_y = x_train[:50], y_train[:50]
    m.optimize(init_x, init_y)

    # Let the model actively learn on its own
    unobserved_x, unobserved_y = x_train[50:530], y_train[50:530]
    active_learn(m, init_x, init_y, unobserved_x, unobserved_y, iters=100)

    # Evaluate our model against test set!
    loss, accuracy = m.evaluate(x_test, y_test)
    print(f'Loss: {loss}') # 0.836
    print(f'Accuracy: {accuracy}') # 0.8219


if __name__ == '__main__':
    main()
