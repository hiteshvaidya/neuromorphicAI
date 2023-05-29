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

    def __str__(self) -> str:
        return str(self.shapeX) + ', ' + str(self.shapeY) + ': ' + str(self.label)