"""
MinLayer.py

description: Layer for performing custom min pooling
version: 1.0
author: Hitesh Vaidya
"""

# import libraries
import tensorflow as tf

class MinLayer(tf.keras.layers.Layer):
    """
    This class forms a custom min pool layer

    :param tf: tensorflow
    :type tf: tensorflow layer object
    """
    def __init__(self, input_shape, strides=(1, 1)):
        """
        constructor

        :param input_shape: size of the layer
        :type input_shape: int
        :param strides: strides for min pooling, defaults to (1, 1)
        :type strides: tuple, optional
        """
        super(MinLayer, self).__init__()
        self.pool_size = input_shape
        self.strides = strides

    def call(self, inputs):
        """
        Gives the coordinates of the minimum value in a matrix

        :param inputs: input matrix
        :type inputs: float32
        :return: coordinates of the minimum values
        :rtype: array of two elements
        """
        output = tf.argmin(tf.reshape(inputs, [-1]), axis=None).numpy()
        return [output //inputs.shape[0], output%inputs.shape[1]]
