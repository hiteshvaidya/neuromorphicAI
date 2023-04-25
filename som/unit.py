"""
This file contains code for each unit of the SOM
"""
from ast import If, Not
from locale import normalize
import math
import pickle
import numpy as np
import tensorflow as tf

class Unit(object):

    def __init__(self, sizeX, sizeY, n_types, initial_running_variance, radius, learning_rate):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.n_types = n_types
        self.class_stats = tf.zeros(n_types)
        self.values = tf.zeros((sizeX, sizeY))
        self.variance = initial_running_variance * tf.ones((sizeX, sizeY))
        self.radius = radius
        self.learning_rate = learning_rate
        self.bmu_count = 0

    def getSize(self):
        pair = (self.sizeX, self.sizeY)
        return pair

    def getClassStats(self):
        return self.class_stats

    def getRadius(self):
        return self.radius

    def getLearningRate(self):
        return self.learning_rate

    def getVariance(self):
        return self.variance

    def decayRadius(self):
        self.radius = tf.maximum(0.00001, self.radius * tf.exp(-self.bmu_count / 15))

    def decayLearningRate(self):
        self.learning_rate = tf.maximum(0.00001, self.learning_rate * tf.exp(-self.bmu_count / 25))

    # seed is np.random.
    def setGeneratorSeed(seed):
        tf.random.set_seed(seed)

    def setCoordinates(self, x, y):
        self.coordinateX = x
        self.coordinateY = y

    def setValues(self, mean, std):
        self.values = tf.random.normal([self.sizeX, self.sizeY], mean, std)
        minValue = tf.reduce_min(self.values)
        maxValue = tf.reduce_max(self.values)
        if not (minValue >= 0 and maxValue < 1):
            self.values /= 255.0

    def getValues(self):
        return self.values

    def generateRandomVector(self, mean, std):
        image = tf.random.normal([self.sizeX, self.sizeY], mean, std)
        return image

    def generateImage(self, mean, std):
        rnd = self.generateRandomVector(mean, std)
        image = tf.zeros([self.sizeX, self.sizeY])
        image = rnd * tf.sqrt(self.variance) + self.values
        return image

    def clampValues(self, lower, upper):
        self.values = tf.clip_by_value(self.values, lower, upper)

    def resetClassStats(self):
        self.class_stats = tf.zeros(n_types)

    def endEpoch(self, number_samples):
        self.variance /= number_samples
