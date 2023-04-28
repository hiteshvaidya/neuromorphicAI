"""
network.py

description: code for SOM
version: 2.0
author: Hitesh Vaidya
"""

# import libraries
import tensorflow as tf
import os
from layer import CustomDistanceLayer
import dataloader
from MinLayer import MinLayer
import util

class Network(tf.keras.Model):
    """
    This class forms the network for the operation of SOM

    :param tf: tensorflow
    :type tf: tensorflow layer object
    """
    def __init__(self, imgSize, n_units, n_classes, radius, learning_rate, running_variance, running_variance_alpha):
        """
        constructor

        :param imgSize: number of pixels in a row and column of the input image or a unit in the SOM
        :type imgSize: int
        :param n_units: number of units in the SOM
        :type n_units: int
        """
        super(Network, self).__init__()
        # Total number of pixels in a row and column of SOM
        self.shapeX = imgSize * n_units
        self.shapeY = imgSize * n_units

        # Total number of units in the SOM
        self.unitsX = n_units
        self.unitsY = n_units

        # randomly initialize pixels of the SOM
        self.som = tf.random.normal([self.shapeX, self.shapeY],
                                    0.1, 0.3)
        # clip values between 0 and 1
        self.som = tf.clip_by_value(self.som, 0.0, 1.0)

        # bmu_count for every unit i.e. how many number of times a unit was selected as BMU
        self.bmu_count = tf.zeros([n_units, n_units, n_classes])

        # Initialize the matrix for running variance
        self.running_variance = tf.ones([self.shapeX, self.shapeY]) * running_variance
        # alpha value for updating running variance
        self.running_variance_alpha = running_variance_alpha

        # Calculate distances between units of the SOM
        self.cartesian_distances = util.calculate_distances(n_units, n_units)
        
        # Create a mask for weight update
        self.mask = tf.zeros([self.shapeX, self.shapeY])

        # Create a matrix for keeping a count of how many times a unit was selected as BMU
        self.class_count = tf.zeros([n_units, n_units])

        # Create a matrix for radius of every unit
        self.radius = radius * tf.ones([n_units, n_units])

        # Create a matrix for learning rate of every unit
        self.learning_rates = learning_rate * tf.ones([n_units, n_units])
        
        # Declare the layers of the network
        self.layer1 = CustomDistanceLayer(imgSize, n_units)
        self.layer2 = MinLayer(n_units)

    def decayRadius(self, bmu):
        decay = tf.exp(-self.bmu_count[bmu] / 15)
        self.radius = tf.tensor_scatter_nd_update(self.radius, bmu, decay)
        self.radius = tf.math.maximum(0.00001 * tf.ones(
                                        tf.shape(self.radius)), 
                                        self.radius)

    def decayLearningRate(self, bmu):
        decay = tf.exp(-self.bmu_count[bmu] / 25)
        self.learning_rates = tf.tensor_scatter_nd_update(self.learning_rates,
                                                    bmu, decay)
        self.learning_rates = tf.math.maximum(0.00001 * tf.ones(
                                        tf.shape(self.learning_rates)), 
                                        self.learning_rates)

    def create_mask(self, bmu, input_matrix):
        # create a distance modifier for neighbourhood function
        distance_modifier = 1.0 / (2.0 * self.radius[bmu[0], bmu[1]] * self.radius[bmu[0], bmu[1]])

        # create a constant used for updating variance_alpha
        constant = -1.0 * tf.math.log(1E-7 / self.learning_rates[bmu[0], bmu[1]]) / distance_modifier

        diff, old_variance, variance_alpha, final_modifier = 0.0, 0.0, 0.0, 0.0

        # We do not perform weight update for units that are farther in the neighbourhood of BMU, than the radius of BMU 
        # This mean, we use the radius of BMU as threashold and set distance values of all the units farther than the threshold to be 0 so as to avoid weight update for them
        modifier = tf.where(self.cartesian_distances[:, :, bmu[0], bmu[1]] > self.radius[bmu[0], bmu[1]], tf.zeros_like(self.radius), self.cartesian_distances[:, :, bmu[0], bmu[1]])

        final_modifier = self.learning_rates * tf.math.exp(-modifier) * distance_modifier

        print("final_modifier: ", (final_modifier))

        final_modifier = tf.repeat(final_modifier, repeats=self.shapeY // self.unitsY, axis=1)
        print("final_modifier: ", tf.shape(final_modifier))
        final_modifier = tf.repeat(final_modifier, repeats=self.shapeX // self.unitsX, axis=0)
        print("final_modifier: ", tf.shape(final_modifier))
        final_modifier = tf.reshape(final_modifier, [self.shapeX, self.shapeY])

        print("final_modifier: ", (final_modifier))

        variance_alpha  = (self.running_variance_alpha - 0.5) + 1.0 / (1.0 + tf.math.exp(-self.cartesian_distances[:, :, bmu[0], bmu[1]] / constant))
        variance_alpha = tf.clip_by_value(variance_alpha, 0.0, 1.0)

        print("variance_alpha:\n", variance_alpha)

        # Perform the weight update
        self.som += final_modifier * (input_matrix - self.som)

        # Update running variance of SOM
        self.running_variance = variance_alpha * self.running_variance + (1.0 - variance_alpha) * (input_matrix - self.som) * (input_matrix - self.som)

        # clip the values of SOM
        self.som = tf.clip_by_value(self.som, 0.0, 1.0)

        # Decay parameters
        self.decayRadius(bmu)
        self.decayLearningRate(bmu)

    
    def forwardPass(self, x):
        """
        Forward pass through the network

        :param x: input image
        :type x: matrix of float values
        """
        input_matrix, z = self.layer1(self.som, self.running_variance, x)
        print("feature map: ", tf.shape(z))
        bmu = self.layer2(z)
        print("input_matrix: \n", input_matrix)
        print("=======================")
        print("min value: ", bmu)
        self.create_mask(bmu, input_matrix)
        

if __name__ == '__main__':
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Declare the object of the network
    network = Network(28, 2, 1.5, 0.7, 0.5, 0.9)
    # Load the data
    (x_train, y_train), (x_test, y_test) = dataloader.loadmnist()
    # Test the forward pass
    print("x_train[0]: \n", x_train[0])
    network.forwardPass(x_train[0])
