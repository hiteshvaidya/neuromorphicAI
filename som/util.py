"""
This file contains code for utility functions
"""

import tensorflow as tf
import numpy as np
import math
from testSOM import testClass
import dataloader
from tqdm import tqdm
import os
# from network import Network


def getTrainingLabel(label, task_size, training_type):
    """
    Get label of the dataset as per training type i.e. 
    [0,1,2,...] - class incremental or 
    [0,1] - domain incremental

    :param label: sample label
    :type label: int
    :param task_size: number of classes in a task/domain
    :type task_size: int
    :param training_type: 'class' / 'domain'
    :type training_type: str
    """
    # convert label to [0,1] if training is domain incremental
    output = None
    if training_type == 'domain':
        output = label % task_size
        # do nothing to the label if training type is class incremental
        return output
    elif training_type == 'class':
        return label

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

def getTaskAccuracy(predictions, labels):
    """
    Given labels and predictions, calculate class-wise accuracy for FWT and BWT

    :param predictions: predictions
    :type predictions: numpy array
    :param labels: labels
    :type labels: numpy array
    """
    # class_count = np.zeros(len(np.unique(labels)), dtype=np.float32)
    # accuracy = np.zeros(len(np.unique(labels)), dtype=np.float32)

    # for i in range(len(labels)):
    #     if labels[i] == predictions[i]:
    #         accuracy[labels[i]] += 1
    #     class_count[labels[i]] += 1

    # # element-wise divide accuracy by class_count
    # accuracy = accuracy / class_count

    accuracy = tf.math.count_nonzero(tf.equal(predictions, labels), 
                                    dtype=tf.int32) 
    accuracy /= tf.shape(labels)[0]

    return accuracy

def getRandomAccuracy(network, test_samples, n_tasks, task_size, training_type):
    # number of classes in complete training
    n_classes = 0
    if training_type == 'class':
        n_classes = n_tasks * task_size
    elif training_type == 'domain':
        n_classes = task_size
        
    test_config = network.getConfig()
    test_model = testClass(test_config['som'], 
                    test_config['shapeX'], 
                    test_config['shapeY'], 
                    test_config['unitsX'], 
                    test_config['unitsY'], 
                    test_config['class_count'], 
                    n_classes)
    test_model.setPMI()
    total_tested_labels = np.zeros(n_classes, dtype=np.float32)
    b = np.zeros(n_tasks, dtype=np.float32)

    for index, samples in enumerate(test_samples):
        outputs = []
        labels = []
        for sample in samples:
            feature_map = test_model.layer1(test_config['som'], 
                                            sample.getImage())
            bmu = test_model.layer2(feature_map)
            output = tf.math.argmax(test_model.get_bmu_PMI(bmu))
            outputs.append(output)
            label = getTrainingLabel(sample.getLabel(), 
                                    task_size,
                                    training_type)
            labels.append(label)
        outputs = tf.convert_to_tensor(outputs, dtype=tf.int32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        b[index] = getTaskAccuracy(outputs, labels)
        
    return b

def getBWT(accuracies):
    accuracies = accuracies.numpy()
    bwt = 0
    for i in range(accuracies.shape[0]-1):
        bwt += accuracies[-1, i] - accuracies[i, i]
    bwt /= (accuracies.shape[0]-1)
    return bwt

def getFWT(accuracies, b):
    accuracies = accuracies.numpy()
    fwt = 0
    for i in range(1, accuracies.shape[0]):
        fwt += accuracies[i-1, i] - b[i]
    fwt /= (accuracies.shape[0] - 1)

    return fwt

def getAverageAccuracy(accuracies):
    accuracies = accuracies.numpy()
    avgAccuracy = 0
    avgAccuracy = float(np.sum(accuracies[-1, :]) / accuracies.shape[0])
    return avgAccuracy 

def getLearningAccuracy(accuracies):
    value = tf.linalg.trace(accuracies) / tf.cast(tf.shape(accuracies)[0], dtype=tf.float32)
    value = value.numpy()
    return value

def getForgettingMeasure(accuracies):
    values = accuracies[-1, :] - tf.reduce_max(accuracies, axis=0)
    forgettingMeasure = tf.reduce_sum(tf.abs(values)) / tf.cast(tf.shape(accuracies)[0], dtype=tf.float32)
    forgettingMeasure = forgettingMeasure.numpy()
    return forgettingMeasure

def dendSOMTaskAccuracy(networks, dataset, patch_size, stride, 
                        n_tasks, task_size, n_soms, training_type):
    # number of classes in complete training
    n_classes = 0
    if training_type == 'class':
        n_classes = n_tasks * task_size
    elif training_type == 'domain':
        n_classes = task_size

    # save the trained model
    test_models = []
    config = None
    for count in range(n_soms):
        config = networks[count].getConfig()
        test_models.append(testClass(config['som'], 
                    config['shapeX'], 
                    config['shapeY'], 
                    config['unitsX'], 
                    config['unitsY'], 
                    config['class_count'], 
                    n_classes))
        test_models[-1].setPMI()

    del networks    
    # tf.scatter_nd_update(labels, [[count]], [test_models[count].getPMI()])
    
    # labels = tf.argmax(labels, axis=0)
    
    # load test samples
    test_samples = dataloader.loadNistTestData("../data/" + dataset,
                                            training_type,
                                            n_tasks,
                                            task_size)    

    final_accuracies = np.zeros(n_tasks, dtype=np.float32)
    tqdm.write("measuring test accuracy")
    for cursor, samples in tqdm(enumerate(test_samples)):
        predictions = tf.Variable([], dtype=tf.int32)

        # collect labels of test_samples
        labels = tf.convert_to_tensor([getTrainingLabel(sample.getLabel(),
                                                    n_classes,
                                                    training_type) 
                                        for sample in samples])
        
        # split every image into patches
        test_samples = dataloader.breakImages(samples, patch_size, stride)

        for sample in test_samples:
            # PMI for BMU from every dendSOM
            pmis = tf.zeros(n_classes)

            # Test every dendSOM on an input test sample
            for count in range(n_soms):
                # forward pass for the test phase
                feature_map = test_models[count].layer1(
                                                config['som'],
                                                sample[count].getImage())
                # Get the best matching unit for test sample
                bmu = test_models[count].layer2(feature_map)
                # Get the PMI of the bmu from the current dendSOM and 
                # add it to store cumulative PMI of every label from every dendSOM
                # pmi shape: [n_soms, n_classes]
                pmis += test_models[count].get_bmu_PMI(bmu)
            # Calculate predicted label by performing argmax over PMI of every label
            # concatenate the predicted label value in `predictions` tensor
            output = tf.math.argmax(pmis, output_type=tf.dtypes.int32)
            predictions = tf.concat([predictions, [output]], axis=0)

        predictions = tf.cast(predictions, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
    
        task_accuracy = getTaskAccuracy(predictions, labels)
        # final_accuracies = np.append(final_accuracies, task_accuracy, axis=0)
        final_accuracies[cursor] = task_accuracy.numpy()

    return final_accuracies