import tensorflow as tf
import util

class CustomDistanceLayer(tf.keras.layers.Layer):
    """
    This is a custom layer for performing distance operation inspired from the convolutional layer in tensorflow

    :param tf: tensorflow object
    :type tf: layer object
    """

    def __init__(self, kernel_size, ) -> None:
        """
        Constructor for the distance layer

        :param kernel_size: size of the kernel
        :type kernel_size: int
        """
        super(CustomDistanceLayer, self).__init__()
        # number of filters
        self.filters = 1
        # size of the kernel
        self.kernel_size = kernel_size
    
    def build(self, som_shape, input_image):
        """
        Assign the input image as a kernel for this layer

        :param input_shape: input shape for the kernel
        :type input_shape: tuple
        :param input_image: input image
        :type input_image: matrix/tensor
        """
        # intialize the kernel
        kernel = tf.tile(input_image, som_shape)
        self.kernel = self.add_weight(shape=kernel.shape,
                                      initializer=tf.keras.initializers.Constant(kernel),
                                      trainable=False)
        
    def call(self, som_matrix, som_running_variances):
        """
        This function calculates the distance between the SOM matrix and the input image using the running variance values. We get an output matrix of the size of [number_of_units, number_of_units] where each index contains the distance value between the input image and the pixels of the corresponding unit

        :param som_matrix: SOM matrix
        :type som_matrix: tensor of float values
        :param som_running_variances: matrix of running variance
        :type som_running_variances: tensor of float values
        :return: matrix of distance values
        :rtype: tensor of float values
        """
        # Obtain a matrix of element-wise distance values using running variance of every pixel in the SOM
        distance_matrix = util.global_variance_distance(self.kernel, 
                                                      som_matrix, som_running_variances)
        
        # Extract patches from the distance_matrix using the size of kernel
        patches = tf.image.extract_patches(images=tf.expand_dimms(
                                                                distance_matrix, axis=0),
                                           sizes=[1, self.kernel_size, 
                                                  self.kernel_size, 1], 
                                           strides=[1, 1, 1, 1], 
                                           rates=[1, 1, 1, 1], 
                                           padding='VALID')

        # Reshape the patches to a 2D tensor where each row corresponds to a patch and each column to a pixel value
        patch_shape = patches.shape
        print("shape of patches: ", patch_shape)
        patches = tf.reshape(patches, [-1, self.kernel_size * self.kernel_size])
        patches = tf.reduce_sum(patches, axis=1)
        output_shape = som_matrix // self.kernel_size
        patches = tf.reshape(patches, [output_shape, output_shape])

        return patches