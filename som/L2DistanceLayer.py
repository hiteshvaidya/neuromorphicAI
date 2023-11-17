import tensorflow as tf
from tensorflow.keras.layers import Layer
import util

class L2DistanceLayer(Layer):
    """
    This is a custom layer for performing distance operation inspired from the convolutional layer in tensorflow

    :param tf: tensorflow object
    :type tf: layer object
    """
    def __init__(self, kernel_size, tile_shape, **kwargs):
        """
        Constructor for the distance layer

        :param kernel_size: size of the kernel or input image
        :type kernel_size: int
        :param tile_shape: number of units/neurons
                            in a row/column of SOM
        :type tile_shape: int
        """
        self.kernel_size = kernel_size
        self.tile_shape = tile_shape
        super(L2DistanceLayer, self).__init__(**kwargs)
    
    def build(self, input_shapes):
        """
        Assign the input image as a kernel for this layer

        :param input_shapes: input shape for the kernel
        :type input_shape: tuple
        """
        self.kernel = self.add_weight(name='kernel', 
                                      shape=[input_shapes[0], input_shapes[1]],
                                      initializer='glorot_uniform',
                                      trainable=False)
        super(L2DistanceLayer, self).build(input_shapes)

    def call(self, som_matrix, input):
        """
        This function calculates the distance between the SOM matrix and the input image using the L2 distance.
        We get an output matrix of the size of [number_of_units, number_of_units] where each index contains the L2 distance value between the input image and the pixels of the corresponding unit.

        :param som_matrix: SOM matrix
        :type som_matrix: tensor of float values
        :param som_running_variances: matrix of running variance
        :type som_running_variances: tensor of float values
        :param input: input image
        :type input: matrix of float values
        :return: matrix of distance values
        :rtype: tensor of float values
        """
        # expand dimensions of distance matrix to [1, num_pixels, num_pixels, 1]
        som_matrix = tf.expand_dims(tf.expand_dims(som_matrix, axis=0), axis=-1)

        # Extract patches from the distance_matrix using the size of kernel
        patches = tf.image.extract_patches(images=som_matrix,
                                           sizes=[1, self.kernel_size, 
                                                  self.kernel_size, 1], 
                                           strides=[1, self.kernel_size, self.kernel_size, 1], 
                                           rates=[1, 1, 1, 1], 
                                           padding='VALID')
        
        # squeeze matrix to remove all the dimensions of size 1
        patches = tf.squeeze(patches)
        # Reshape the patches to a 2D tensor where each row corresponds to a patch and each column to a pixel value        
        patches = tf.reshape(patches, [-1, self.kernel_size * self.kernel_size])
        
        input = tf.reshape(input, [1, -1])
        patches = util.l2_distance(patches, input)

        # reshape patches to [num_units_in_SOM, num_units_in_SOM]
        patches = tf.reshape(patches, [self.tile_shape, self.tile_shape])

        return patches
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] # Output shape is (batch_size, )
    
if __name__ == "__main__":
    input1 = tf.keras.layers.Input(shape=(4, 4, 3))
    input2 = tf.keras.layers.Input(shape=(4, 4, 3))

    l2_distance_layer = L2DistanceLayer()([input1, input2])

    model = tf.keras.models.Model(inputs=[input1, input2], outputs=l2_distance_layer)
    