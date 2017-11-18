import os

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from layers.inference_dropout import InferenceDropout as Dropout


np.random.seed(42)

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
        self._saved = None
        self._loss = loss
        self._optimizer = optimizer
        self.init_model()


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

        self._model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

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

    def init_model(self):
        if self._saved:
            K.clear_session()
            self.load_model(self._saved)
        else:
            model = Sequential()

            # Conv1
            model.add(Conv2D(20,
                             kernel_size=(5, 5),
                             input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # Conv2
            model.add(Conv2D(50, (5, 5)))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # Fully connected
            model.add(Flatten())
            model.add(Dense(500, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(loss=self._loss,
                          optimizer=self._optimizer,
                          metrics=['accuracy'])

            self._model = model
