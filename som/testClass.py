"""
testClass.py

description: Use CosineDistanceLayer for inference
version: 1.0
author: Manali Dangarikar

"""

# import libraries
import tensorflow as tf
import os
from CosineDistanceLayer import CosineDistanceLayer
from MaxLayer import MaxLayer
import dataloader


class testClass(tf.keras.Model):
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
        super(testClass, self).__init__()
        self.shapeX = imgSize * n_units
        self.shapeY = imgSize * n_units
        # randomly initialize pixels of the SOM
        self.som = tf.random.normal([self.shapeX, self.shapeY],
                                    0.1, 0.3)
        # clip values between 0 and 1
        self.som = tf.clip_by_value(self.som, 0.0, 1.0)

        # Initialize the matrix for predicted class
        self.predicted_class = tf.ones([n_units, n_units]) * (-1.0)
        
        # Declare the layers of the network
        self.layer1 = CosineDistanceLayer(imgSize, n_units)
        self.layer2 = MaxLayer(n_units)

    def init(self, som, predicted_class):
        self.som = som
        self.predicted_class = predicted_class

    def InferencePass(self, x):
        """
        Inference pass through the network

        :param x: input image
        :type x: matrix of float values
        """
        x = self.layer1(self.som, x)
        x = self.layer2(x)
        print("max value:", x)
        
    def getPredictedClass(self, x):
        predictedClass = tf.gather(tf.gather(self.predicted_class, x[0]), x[1])
        return predictedClass

    def getAccuracy(self, y_pred, y_test):
        correct_predictions = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_test), tf.float32))
        accuracy = correct_predictions / tf.cast(tf.shape(y_test)[0], tf.float32)
        return accuracy


if __name__ == '__main__':
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load the data
    (x_train, y_train), (x_test, y_test) = dataloader.loadmnist()

    # Declare object of test class using som matrix and predicted_class created during training
    test = testClass(som, predicted_class)

    # Infer for x_test[0]
    # step 1: get the best matching unit for the input test sample
    bmu = test.InferencePass(x_test[0])

    # step 2: get the corresonding predicted class for the above bmu
    y_pred = tf.constant(-1, shape=y_test.shape)
    y_pred = tf.tensor_scatter_nd_update(y_pred, [[0]], [test.getPredictedClass(bmu)])

    # step 3: calculate accuracy
    runningAccuracy = test.getAccuracy(y_pred, y_test)
