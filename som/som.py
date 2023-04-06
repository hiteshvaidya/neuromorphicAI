"""
This file contains code for the self organizing map

Version: 1.0
author: Hitesh Vaidya
"""

from Unit import Unit
from Sample import Sample
import numpy as np
import os
import tensorflow as tf
import random
import util

class SOM(object):

    def __init__(self, shapeX, shapeY, valueSizeX, valueSizeY, radius, learning_rate, nClasses, mean, std, running_variance_alpha,  initial_running_variance):
        """
        constructor for the self organizing map

        :param shapeX: number of rows of units in the SOM
        :type shapeX: int
        :param shapeY: number of columns of units in the SOM
        :type shapeY: int
        :param valueSizeX: number of rows of pixels in every unit of the SOM
        :type valueSizeX: int
        :param valueSizeY: number of columns of pixels in every unit of the SOM
        :type valueSizeY: int
        :param radius: radius of every unit in the SOM
        :type radius: float
        :param learning_rate: learning rate of every unit in the SOM
        :type learning_rate: float
        :param nClasses: number of input classes
        :type nClasses: int
        :param mean: mean value for generating initial pixel values of every unit in the SOM
        :type mean: float
        :param std: std value for generating intial pixel values of every unit in the SOM
        :type std: float
        :param running_variance_alpha: alpha value for upating running variance 
        :type running_variance_alpha: float
        :param initial_running_variance: initial value of running variance
        :type initial_running_variance: float
        """
        # number of rows in the SOM
        self.shapeX = shapeX
        # number of columns in the SOM
        self.shapeY = shapeY
        # alpha for updating running variance
        self.running_variance_alpha = running_variance_alpha
        # 2D array of units in the SOM
        self.units = np.array([np.array([Unit(valueSizeX, valueSizeY, nClasses, initial_running_variance, radius, learning_rate) for j in range(shapeY)]) for i in range(shapeX)])

        # set the parameters of every unit in the SOM
        for i in range(shapeX):
            for j in range(shapeY):
                # assign seed value to random object of the unit
                self.units[i, j].setGeneratorSeed(i*shapeX + j)
                # every unit is placed at coordinates (i,j) in a euclidean plane
                self.units[i, j].setCoordinates(i, j)
                # set mean and std for generating inital values in a unit
                self.units[i, j].setValues(mean, std)

        # calculate euclidean distance between all the units and store them in a 4D array
        self.cartesian_distances = util.calculate_distances(shapeX, shapeY)

        # convert numpy array of objects to tensor
        self.units = tf.convert_to_tensor(self.units)


    def get_number_of_units(self):
        """
        get shape of the SOM i.e. number of rows and columns of units in the SOM

        :return: number of rows and columns
        :rtype: tuple
        """
        return (self.shapeX, self.shapeY)
    
    def getUnit(self, row, col):
        """
        Accessor for unit at specified coordinates

        :param row: row number
        :type row: int
        :param col: column number
        :type col: int
        :return: unit at [row, col]
        :rtype: Unit() object
        """
        return self.units[row, col]
    

    def reset_class_stats(self, selected_targets):
        """
        reset class stats of every unit in the SOM

        :param selected_targets: array of statistics for every class
        :type selected_targets: array
        """
        for r in range(self.shapeX):
            for c in range(self.shapeY):
                self.units[r, c].reset_class_stats(selected_targets)

    def end_epoch(self, number_samples):
        """
        consolidate statistics at the end of epoch for every unit

        :param number_samples: number of samples in the epoch
        :type number_samples: int
        """
        for r in range(self.shapeX):
            for c in range(self.shapeY):
                self.units[r, c].end_epoch(number_samples)
    
    def insert_sample(self, current_iteration, sample, distance_choice):
        compare_distance = 0
        if (distance_choice == "cosine"):
            compare_distance = 