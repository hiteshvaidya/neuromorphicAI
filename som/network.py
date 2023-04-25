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

class Network(tf.keras.Model):
    """
    This class forms the network for the operation of SOM

    :param tf: tensorflow
    :type tf: tensorflow layer object
    """
    def __init__(self, imgSize, n_units):
        """
        constructor

        :param imgSize: number of pixels in a row and column of the input image or a unit in the SOM
        :type imgSize: int
        :param n_units: number of units in the SOM
        :type n_units: int
        """
        super(Network, self).__init__()
        self.shapeX = imgSize * n_units
        self.shapeY = imgSize * n_units
        # randomly initialize pixels of the SOM
        self.som = tf.random.normal([self.shapeX, self.shapeY],
                                    0.1, 0.3)
        # clip values between 0 and 1
        self.som = tf.clip_by_value(self.som, 0.0, 1.0)
        # Initialize the matrix for running variance
        self.running_variance = tf.ones([self.shapeX, self.shapeY]) * 0.5
        
        # Declare the layers of the network
        self.layer1 = CustomDistanceLayer(imgSize, n_units)
        self.layer2 = MinLayer(n_units)
    
    def forwardPass(self, x):
        """
        Forward pass through the network

        :param x: input image
        :type x: matrix of float values
        """
        x = self.layer1(self.som, self.running_variance, x)
        x = self.layer2(x)
        print("min value:", x)
        

if __name__ == '__main__':
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Declare the object of the network
    network = Network(28, 5)
    # Load the data
    (x_train, y_train), (x_test, y_test) = dataloader.loadmnist()
    # Test the forward pass
    network.forwardPass(x_train[0])
