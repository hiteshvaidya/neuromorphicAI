

# import libraries
import numpy as np
import tensorflow as tf

class Sample(object):   
    """
    class for representing one input sample
    """

    def __init__(self, label, sizeX, sizeY, image):
        """
        constructor

        :param label: label of the image
        :type label: int
        :param sizeX: number of rows of pixels
        :type sizeX: int
        :param sizeY: number of columns of pixels
        :type sizeY: int
        :param image: image pixel matrix
        :type image: float32
        """
        self.label = label
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.image = tf.convert_to_tensor(image, dtype=tf.float32)

    def printSample(self):
        """
        print this sample
        """
        print("label:", self.label)
        print("sizeX:", self.sizeX)
        print("sizeY:", self.sizeY)
        print("values:")
        for row in self.values.numpy():
            print(row)

    def getStats(self):
        """
        get statistics of this sample

        :return: mean and std
        :rtype: tuple
        """
        mean = tf.reduce_mean(self.values)
        std = tf.math.reduce_std(self.values)
        return (mean, std)

    def getSum(self):
        """
        get sum of all pixels in this sample

        :return: sum of pixels
        :rtype: float32
        """
        total = tf.reduce_sum(self.values)
        return (total)

    def getValues(self):
        """
        Accesor for values of image

        :return: image pixel values
        :rtype: float32
        """
        return (self.values)

    def getLabel(self):
        """
        label of the image

        :return: label
        :rtype: int
        """
        return (self.label)

    def setLabel(self, label):
        """
        Mutator for label of the sample

        :param label: image label
        :type label: int
        """
        self.label = label

    def write(self, file):
        """
        write this sample to a file

        :param file: filename
        :type file: str
        """
        file.write(str(self.label) + "\n")
        file.write(str(self.sizeX) + "\n")
        file.write(str(self.sizeY) + "\n")
        for row in self.values.numpy():
            file.write(" ".join([str(val) for val in row]) + "\n")

    def read(self, file):
        """
        Read a sample from a file

        :param file: filename
        :type file: str
        """
        self.label = int(file.readline())
        self.sizeX = int(file.readline())
        self.sizeY = int(file.readline())
        self.values = []
        for _ in range(self.sizeX):
            row = [float(val) for val in file.readline().split()]
            self.values.append(row)
        self.values = tf.convert_to_tensor(self.values, dtype=tf.float32)

    def RMS_value(self, s):
        """
        distance of this sample from a given matrix

        :param s: given matrix
        :type s: float32
        :return: RMS distance value
        :rtype: float32
        """
        output = 0.0
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                output += tf.math.square(self.values[i][j] - s.values[i][j])

        output /= self.sizeX*self.sizeY
        output = tf.math.sqrt(output)
        return output.numpy()

"""
sample.py

description: This file contains code for loading one input sample
version: 1.0
author: Manali Dangarikar
"""

import numpy as np
import tensorflow as tf

class Sample(object):
    """
    Sample class to store input images

    :param object: Object
    :type object: Object class
    """

    def __init__(self, label, shapeX, shapeY, image):
        """
        constructor

        :param label: class of the image
        :type label: int
        :param shapeX: shapeX of the image
        :type shapeX: int
        :param shapeY: shapeY of the image
        :type shapeY: int
        :param image: image pixels
        :type image: numpy 2D array
        """
        self.label = label
        self.shapeX = shapeX
        self.shapeY = shapeY
        self.image = image

    def getLabel(self):
        """
        Accessor for label

        :return: self.label
        :rtype: int
        """
        return self.label
    
    def setLabel(self, y):
        self.label = y
    
    def getImage(self):
        """
        Accessor for the image pixels

        :return: self.image
        :rtype: numpy 2D array
        """
        return tf.convert_to_tensor(self.image, dtype=tf.float32)
    
    def setImage(self, image):
        """
        Mutator for image field

        :param image: new image
        :type image: numpy array
        """
        self.image = image
    
    def getShape(self):
        """
        Accessor for shape of sample

        :return: self.shapeX, self.shapeY
        :rtype: tuple
        """
        return (self.shapeX, self.shapeY)