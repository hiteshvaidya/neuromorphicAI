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
from sample import Sample
import dataloader
from MinLayer import MinLayer
import util
import cv2
import argparse

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
        self.class_count = tf.zeros([n_units, n_units, n_classes])

        # Initialize the matrix for running variance
        self.running_variance = tf.ones([self.shapeX, self.shapeY]) * running_variance
        # alpha value for updating running variance
        self.running_variance_alpha = running_variance_alpha

        # Calculate distances between units of the SOM
        self.cartesian_distances = util.calculate_distances(n_units, n_units)
        
        # Create a mask for weight update
        self.mask = tf.zeros([self.shapeX, self.shapeY])

        # Create a matrix for keeping a count of how many times a unit was selected as BMU
        # self.class_count = tf.zeros([n_units, n_units])

        # Create a matrix for radius of every unit
        self.radius = radius * tf.ones([n_units, n_units])

        # Create a matrix for learning rate of every unit
        self.learning_rates = learning_rate * tf.ones([n_units, n_units])
        
        # Declare the layers of the network
        self.layer1 = CustomDistanceLayer(imgSize, n_units)
        self.layer2 = MinLayer(n_units)

    def decayRadius(self, bmu):
        """
        Decay current radius value of the best matching unit

        :param bmu: best matching unit
        :type bmu: tuple
        """
        # 15 is a constant value (can be changed)
        decay = tf.exp(-tf.reduce_sum(self.class_count[bmu[0], bmu[1], :]) / 15)
        self.radius = tf.tensor_scatter_nd_update(self.radius, [bmu], [decay])
        self.radius = tf.math.maximum(0.00001 * tf.ones(
                                        tf.shape(self.radius)), 
                                        self.radius)

    def decayLearningRate(self, bmu):
        """
        Decay current learning rate of the best matching unit

        :param bmu: best matching unit
        :type bmu: tuple
        """
        # 25 is a constant value (can be changed)
        decay = tf.exp(-tf.reduce_sum(self.class_count[bmu[0], bmu[1], :]) / 25)
        self.learning_rates = tf.tensor_scatter_nd_update(self.learning_rates,
                                                    [bmu], [decay])
        self.learning_rates = tf.math.maximum(0.00001 * tf.ones(
                                        tf.shape(self.learning_rates)), 
                                        self.learning_rates)
        
    def visualize_model(self):
        """
        Visualize the current state of SOM
        """
        # Display current state of SOM
        cv2.imshow("Self Organizing Map", self.som.numpy())
        # wait for 1 milli-seconds
        cv2.waitKey(1)


    def weight_update(self, bmu, input_matrix, label):
        # create a distance modifier for neighbourhood function
        distance_modifier = 1.0 / (2.0 * self.radius[bmu[0], bmu[1]] * self.radius[bmu[0], bmu[1]])

        # create a constant used for updating variance_alpha
        constant = -1.0 * tf.math.log(1E-7 / self.learning_rates[bmu[0], bmu[1]]) / distance_modifier

        diff, old_variance, variance_alpha, final_modifier = 0.0, 0.0, 0.0, 0.0

        # We do not perform weight update for units that are farther in the neighbourhood of BMU, than the radius of BMU 
        # This mean, we use the radius of BMU as threashold and set distance values of all the units farther than the threshold to be 0 so as to avoid weight update for them
        modifier = tf.where(self.cartesian_distances[ :, :, bmu[0], bmu[1]] > self.radius[bmu[0], bmu[1]], tf.zeros_like(self.cartesian_distances[ :, :, bmu[0], bmu[1]]), self.cartesian_distances[ :, :, bmu[0], bmu[1]])
        modifier = tf.tensor_scatter_nd_update(modifier, [bmu], [1])

        final_modifier = modifier * self.learning_rates * tf.math.exp(-self.cartesian_distances[:, :, bmu[0], bmu[1]] * distance_modifier)

        final_modifier = tf.repeat(final_modifier, repeats=self.shapeY // self.unitsY, axis=1)
        final_modifier = tf.repeat(final_modifier, repeats=self.shapeX // self.unitsX, axis=0)
        final_modifier = tf.reshape(final_modifier, [self.shapeX, self.shapeY])

        # Perform the weight update
        self.som += final_modifier * (input_matrix - self.som)

        # clip the values of SOM
        self.som = tf.clip_by_value(self.som, 0.0, 1.0)

        self.class_count = tf.tensor_scatter_nd_update(self.class_count, [[bmu[0], bmu[1], label]], [self.class_count[bmu[0], bmu[1], label] + 1])

        variance_alpha  = (self.running_variance_alpha - 0.5) + 1.0 / (1.0 + tf.math.exp(-self.cartesian_distances[:, :, bmu[0], bmu[1]] / constant))
        variance_alpha = tf.clip_by_value(variance_alpha, 0.0, 1.0) * modifier
        variance_alpha = tf.repeat(variance_alpha, repeats=self.shapeX // self.unitsX, axis=1)
        variance_alpha = tf.repeat(variance_alpha, repeats=self.shapeY // self.unitsY, axis=0)
        variance_alpha = tf.reshape(variance_alpha, [self.shapeX, self.shapeY])
        # variance_alpha = tf.tile(variance_alpha, [self.shapeX // self.unitsX, self.shapeY // self.unitsY])
        
        # Update running variance of SOM
        self.running_variance = variance_alpha * self.running_variance + (1.0 - variance_alpha) * (input_matrix - self.som) * (input_matrix - self.som)

        # Decay parameters
        self.decayRadius(bmu)
        self.decayLearningRate(bmu)

    
    def forwardPass(self, x, y):
        """
        Forward pass through the network

        :param x: input image
        :type x: matrix of float values
        """
        tiled_input, unit_map = self.layer1(self.som, self.running_variance, x)
        bmu = self.layer2(unit_map)
        self.weight_update(bmu, tiled_input, y)

if __name__ == '__main__':

    # define parser for command line arguments
    parser = argparse.ArgumentParser()

    # add command line arguments
    parser.add_argument('-g', '--gpuid', type=str, default=None, help='gpu id')
    parser.add_argument('-u', '--units', type=int, required=True, default=10, help='number of units in a row of the SOM')
    parser.add_argument('-r', '--radius', type=float, required=True, default=None, help='initial radius of every unit in SOM')
    parser.add_argument('-lr', '--learning_rate', type=float, required=True, default=None, help='initial learning rate of every unit in SOM')
    parser.add_argument('-va', '--variance_alpha', type=float, required=False, default=0.9, help='initial value of alpha for running variance')
    parser.add_argument('-v', '--variance', type=float, required=False, default=0.5, help='initial value of running variance')
    parser.add_argument('-fp', '--filepath', type=str, required=True, default=None, help='filepath for saving trained SOM model')
    args = parser.parse_args()

    # check if gpu exist
    if tf.test.is_gpu_available():
        # set gpu id
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
   
    # Declare the object of the network
    network = Network(28, args.units, 10, args.radius, args.learning_rate, args.variance, args.variance_alpha)
    
    # Perform the forward pass
    for index in range(10):
        # Load the data
        class_train_samples = dataloader.loadClassIncremental("../data/mnist/train/", index, 1)
        
        # train on current class
        for index, sample in enumerate(class_train_samples):
            # print("sample: ", sample)
            # print("sample value: ", sample.values)
            network.forwardPass(sample.getImage(), sample.getLabel())
            network.visualize_model()
            if index == 100:
                break
    
    # Destroy all cv2 windows
    cv2.destroyAllWindows()
    
    if not os.path.exists("logs"):
        os.makedirs("logs")
    # save the trained model
    dataloader.saveModel(network, os.path.join("logs", args.filepath))
