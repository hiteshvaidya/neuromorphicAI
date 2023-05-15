import tensorflow as tf
import util

class CosineDistanceLayer(tf.keras.layers.Layer):
    """
    This is a custom layer for performing distance operation inspired from the convolutional layer in tensorflow

    :param tf: tensorflow object
    :type tf: layer object
    """

    def __init__(self, kernel_size, tile_shape, **kwargs) -> None:
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
        super(CosineDistanceLayer, self).__init__(**kwargs)
    
    def build(self, input_shapes):
        """
        Assign the input image as a kernel for this layer

        :param input_shapes: input shape for the kernel
        :type input_shape: tuple
        """
        # intialize the kernel
        self.kernel = self.add_weight(name='kernel',
                                    shape=[input_shapes[0], input_shapes[1]],
                                    initializer='glorot_uniform',
                                    trainable=False)
        super(CosineDistanceLayer, self).build(input_shapes)
        
    def call(self, som_matrix, input):
        """
        This function calculates the distance between the SOM matrix and the input image using the cosine similarity.
        We get an output matrix of the size of [number_of_units, number_of_units] where each index contains the cosine similarity value between the input image and the pixels of the corresponding unit.

        :param som_matrix: SOM matrix
        :type som_matrix: tensor of float values
        :param som_running_variances: matrix of running variance
        :type som_running_variances: tensor of float values
        :param input: input image
        :type input: matrix of float values
        :return: matrix of distance values
        :rtype: tensor of float values
        """
        # expand dimensions of distance matrix to [1, num_pixels, num_pixles, 1]
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
        patches = util.cosine_similarity(patches, input)

        # reshape patches to [num_units_in_SOM, num_units_in_SOM]
        patches = tf.reshape(patches, [self.tile_shape, self.tile_shape])
        
        return patches