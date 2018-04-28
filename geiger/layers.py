from keras.layers import Layer


class ConsumeMask(Layer):
    """Layer that prevents mask propagation.
    Stolen from https://github.com/raghakot/keras-text/blob/master/keras_text/models/layers.py
    """

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x
