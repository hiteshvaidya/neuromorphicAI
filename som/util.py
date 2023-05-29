"""
This file contains code for utility functions
"""

import tensorflow as tf
import numpy as np
import math

def calculate_distances(rows, cols):
    """
    Calculate distances between coordinates in given range.
    example, given [3,3] matrix, calculate distance between every [i,j] where i and j are in range [0,2].
    This leads to a 4D matrix storing distance between [x,y] and [i,j] where x,i are in the range [0,rows-1] and y,j are in range [0, cols-1]

    :param distances: matrix that stores distance values
    :type distances: 4D numpy matrix
    :return: distances
    :rtype: 4D numpy array
    """
    distances = np.zeros([rows, cols, rows, cols], dtype=np.float32)
    for x in range(rows):
        for y in range(cols):
            for i in range(rows):
                for j in range(cols):
                    distances[x][y][i][j] = (x-i) * (x-i) + (y-j) * (y-j)
    distances = tf.math.sqrt(tf.convert_to_tensor(distances, dtype=tf.float32))
    return distances

def cosine_similarity(tensor1, tensor2):
    """
    calculate row-wise cosine similarity between two tensors

    :param tensor1: first tensor
    :type tensor1: tf.float32
    :param tensor2: second tensor
    :type tensor2: tf.float32
    :return: cosine similarity value
    :rtype: tf.float32
    """
    # Normalize matrix 1 and matrix 2
    normalized_matrix1 = tf.linalg.normalize(tensor1, axis=1)[0]
    normalized_matrix2 = tf.linalg.normalize(tensor2, axis=1)[0]
    # calculate dot product of two tensors
    dot_product = tf.reduce_sum(tf.multiply(normalized_matrix1, normalized_matrix2), axis=1)
    return dot_product

def variance_distance(sample, som_unit_values, som_unit_variances):
    """
    calculate RMS/L2 distance where each squared term is divided by the sqrt of the running variance each pixel in the SOM unit

    :param sample: input sample
    :type sample: 2D matrix of float values
    :param som_unit_values: pixel values of the SOM unit
    :type som_unit_values: 2D matrix of float values
    :param som_unit_variances: running variances of every pixel in a SOM unit
    :type som_unit_variances: 2D matrix of float values
    :return: variance_distance value
    :rtype: float
    """
    distance = tf.reduce_sum((sample - som_unit_values) * (sample - som_unit_values) / tf.math.sqrt(som_unit_variances))

    return distance

def global_variance_distance(input_matrix, som_matrix, som_running_variances):
    """
    calculate the element-wise distance using running variance values but do not perform summation over terms in the L2 distance

    :param input_matrix: tiled matrix of input image
    :type input_matrix: tensor of float values
    :param som_matrix: som matrix
    :type som_matrix: tensor of float values
    :param som_running_variances: matrix of running variances in SOM
    :type som_running_variances: tensor of float values
    :return: matrix of element-wise distance values
    :rtype: tensor of float values
    """
    distance_matrix = (input_matrix - som_matrix) * (input_matrix - som_matrix) / tf.math.sqrt(som_running_variances)
    
    return distance_matrix

def getAccuracy(y_pred, y_test):
    """
    Given the predicted and expected labels, calculate the accuracy

    :param y_pred: predicted labels from the SOM
    :type y_pred: 1D tensor
    :param y_test: expected labels from the SOM
    :type y_test: 1D tensor
    :return: accuracy value
    :rtype: float
    """
    correct_predictions = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_test), tf.float32))
    accuracy = correct_predictions / tf.cast(tf.shape(y_test)[0], tf.float32)
    return accuracy