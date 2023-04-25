"""
CustomArgMaxPoolingLayer.py

description: layer for custom max pooling
version: 1.0
author: Manali Dangarikar 
"""

# import libraries
import tensorflow as tf

class CustomArgmaxPoolingLayer(tf.keras.layers.Layer):
    """
    This class contains code for performing max pooling over distance values

    :param tf: tensorflow
    :type tf: tensorflow layer object
    """
    def __init__(self, input_shape, strides=(1, 1)):
        """
        constructor

        :param input_shape: shape of the input matrix
        :type input_shape: int
        :param strides: strides for max pooling, defaults to (1, 1)
        :type strides: tuple, optional
        """
        super(CustomArgmaxPoolingLayer, self).__init__()
        self.pool_size = input_shape
        self.strides = strides

    def call(self, inputs):
        """
        Gives coordinates of the unit with maximum value

        :param inputs: input matrix
        :type inputs: float32
        :return: coordinates of the unit with maximum value
        :rtype: list
        """
        output = tf.argmax(tf.reshape(inputs, [-1]), axis=None).numpy()
        print('output=',output)
        return [output //inputs.shape[0], output%inputs.shape[1]]
