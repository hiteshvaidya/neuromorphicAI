import tensorflow as tf
import util

class CustomDistanceLayer(tf.keras.layers.Layer):
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
        super(CustomDistanceLayer, self).__init__(**kwargs)
    
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
        super(CustomDistanceLayer, self).build(input_shapes)
        
    def call(self, som_matrix, som_running_variances, kernel):
        """
        This function calculates the distance between the SOM matrix and the input image using the running variance values. We get an output matrix of the size of [number_of_units, number_of_units] where each index contains the distance value between the input image and the pixels of the corresponding unit

        :param som_matrix: SOM matrix
        :type som_matrix: tensor of float values
        :param som_running_variances: matrix of running variance
        :type som_running_variances: tensor of float values
        :param kernel: input image
        :type kernel: matrix of float values
        :return: matrix of distance values
        :rtype: tensor of float values
        """
        # Repeat the input image as per the number of units in a single row or column of the SOM to form one single matrix of the size of the SOM
        self.kernel = tf.tile(kernel, [self.tile_shape, self.tile_shape])

        # Obtain a matrix of element-wise distance values using running variance of every pixel in the SOM
        distance_matrix = util.global_variance_distance(self.kernel, 
                                                      som_matrix, som_running_variances)
        
        # expand dimensions of distance matrix to [1, num_pixels, num_pixles, 1]
        distance_matrix = tf.expand_dims(tf.expand_dims(distance_matrix, axis=0), axis=-1)
        # Extract patches from the distance_matrix using the size of kernel
        patches = tf.image.extract_patches(images=distance_matrix,
                                           sizes=[1, self.kernel_size, 
                                                  self.kernel_size, 1], 
                                           strides=[1, self.kernel_size, self.kernel_size, 1], 
                                           rates=[1, 1, 1, 1], 
                                           padding='VALID')
        
        # squeeze matrix to remove all the dimensions of size 1
        patches = tf.squeeze(patches)
        # Reshape the patches to a 2D tensor where each row corresponds to a patch and each column to a pixel value        
        patches = tf.reshape(patches, [-1, self.kernel_size * self.kernel_size])
        # Add pixel-wise distance for every unit
        patches = tf.reduce_sum(patches, axis=1)
        # reshape patches to [num_units_in_SOM, num_units_in_SOM]
        patches = tf.reshape(patches, [self.tile_shape, self.tile_shape])
        
        return self.kernel, patches