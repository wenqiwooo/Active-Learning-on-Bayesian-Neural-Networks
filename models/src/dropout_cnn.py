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

    def optimize(self, x, y, epochs=12, batch_size=128):
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
        for i in range(num_samples):
            print(f'Collecting sample: {i+1}')
            probs += [self._model.predict(x, verbose=1)]

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

if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] not in ('test', 'train'):
        print('Supply either train/evaluate as an argument to the script.')
        print('e.g. python <script_name>.py train')
        sys.exit()

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

    if sys.argv[1] == 'train':
        m.optimize(x_train, y_train)

        # Saves model
        if len(sys.argv) == 4:
            save_dir, filename = sys.argv[2], sys.argv[3]
            m.save_model(save_dir, filename)
    else:
        if len(sys.argv) < 4:
            print('Please include filepath of saved model for test, and\
                   directory to store predictive mean and variance')
            sys.exit()

        filename = sys.argv[2]
        m.load_model(filename)
        loss, accuracy = m.evaluate(x_test, y_test)

        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')

        # Gets the predictive mean and variance, saves it
        mean, var = m.sample(x_test)

        timestamp = '{:%Y%m%d-%H:%M:%S}'.format(datetime.datetime.now())
        np.save(f'{sys.argv[3]}/mnist-mean-{timestamp}.npy', mean)
        np.save(f'{sys.argv[3]}/mnist-var-{timestamp}.npy', var)
