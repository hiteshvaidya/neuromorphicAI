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
    distances = tf.zeros([rows, cols, rows, cols], dtype=tf.float32)
    for x in range(rows):
        for y in range(cols):
            for i in range(rows):
                for j in range(cols):
                    distances[x][y][i][j].assign(tf.sqrt((x-i) * (x-i) + (y-j) * (y-j)))
                    
    return distances

def cosine_similarity(tensor1, tensor2):
    """
    calculate cosine similarity between two tensors

    :param tensor1: first tensor
    :type tensor1: tf.float32
    :param tensor2: second tensor
    :type tensor2: tf.float32
    :return: cosine similarity value
    :rtype: tf.float32
    """
    # calculate dot product of two tensors
    dot_product = tf.reduce_sum(tf.multiply(tensor1, tensor2))

    # calculate l2 norm of two tensors
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(tensor2)))

    # calculate cosine similarity
    cosine_similarity = dot_product / (norm1 * norm2)

    return cosine_similarity
