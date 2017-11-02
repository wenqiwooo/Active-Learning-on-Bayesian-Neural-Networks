from keras import backend as K
from keras.layers import Dropout

# Define a custom Dropout layer that applies dropout even on inference
class InferenceDropout(Dropout):
    def call(self, inputs, training=None):
        """
        Overrides the default InferenceDropout layer's call function.
        """
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)

        return inputs
