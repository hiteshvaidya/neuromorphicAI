import tensorflow as tf

class CustomArgmaxPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape, strides=(1, 1)):
        super(CustomArgmaxPoolingLayer, self).__init__()
        self.pool_size = input_shape
        self.strides = strides

    def call(self, inputs):
        output = tf.argmax(tf.reshape(inputs, [-1]), axis=None).numpy()
        print('output=',output)
        return [output //inputs.shape[0], output%inputs.shape[1]]
