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
import math

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
    
    def getBMU(self, sample, distance_choice):
        row_index, col_index = -1, -1
        if (distance_choice == "cosine"):
            # get cosine similarity of sample from every unit
            distances = tf.convert_to_tensor(np.array([np.array([self.units[i, j].getCosineSimilarity(sample) for j in range(self.shapeY)]) for i in range(self.shapeX)]))
            
            # get index of maximum cosine similarity value from flattened tensor of distances
            max_index = tf.argmax(tf.reshape(distances, [-1]))

            # get the row and column coordinates of the maximum element in the distances tensor
            row_index = max_index // sample.shape[1]
            col_index = max_index % sample.shape[1]

        elif (distance_choice == "variance"):
            # get variance distance values of input sample from every som unit
            distances = tf.convert_to_tensor(np.array([np.array([self.units[i,j].getVarianceDistance(sample) for j in range(self.shapeY)]) for i in range(self.shapeX)]))

            # get index of the mininum variance distance value from the flattened tensor of distances
            min_index = tf.argmin(tf.reshape(distances, [-1]))

            # get the row and column coordinates of the maximum element in the distances tensor
            row_index = min_index // sample.shape[1]
            col_index = min_index % sample.shape[1]
        
        return (row_index, col_index)

    def insert_sample(self, current_iteration, sample, distance_choice):
        
        # get coordinates of BMU from the SOM based on the choice of distance function
        (bmu_row, bmu_col) = self.getBMU(sample, distance_choice)

        # increase the bmu count (i.e., number of times a unit was selected as BMU) for the newly calculated BMU [bmu_row, bmu_col]
        self.units[bmu_row, bmu_col].class_stats[sample.getLabel()] += 1
        self.units[bmu_row, bmu_col].bmu_count += 1

        distance_modifier = 1.0 / (2.0 * self.units[bmu_row][bmu_col].radius * self.units[bmu_row][bmu_col].radius)

        constant = -1.0 * tf.log(0.0000001 / self.units[bmu_row][bmu_col].learning_rate) / distance_modifier

        diff, old_variance, variance_alpha, final_modifier = 0.0, 0.0, 0.0, 0.0

        for r in range(self.shapeX):
            for c in range(self.shapeY):
                final_modifier = self.units[bmu_row, bmu_col].learning_rate * math.exp(-self.cartesian_distances[r, c, bmu_row, bmu_col]) * distance_modifier

                if (self.cartesian_distances[r, c, bmu_row, bmu_col] > self.units[bmu_row, bmu_col].radius):
                    continue

                variance_alpha = max(0, min(1.0, self.running_variance_alpha - 0.5) + 1 / (1 + math.exp(-self.cartesian_distances[r,c,bmu_row, bmu_col] / constant)))

                for m in range(self.units[r, c].sizeX):
                    for n in range(self.units[r, c].sizeY):
                        diff = (sample.values[m,n] - self.units[r,c].getValues()[m,n])

                        self.units[r, c].values[m, n] += final_modifier * diff

                        self.units[r, c].variance[m, n] = (variance_alpha) * self.units[r, c].variance[m, n] + (1.0 - variance_alpha) * (sample.values[m, n] - self.units[r, c].values[m, n]) * (sample.values[m, n] - self.units[r, c].values[m, n])

                tf.clip_by_value(self.units[r, c], 0.0, 1.0)

        self.units[bmu_row, bmu_col].decayRadius()
        self.units[bmu_row, bmu_col].decayLearningRate()

