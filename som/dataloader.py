"""
dataloader.py

description: This file contains code for dataloader
version: 1.0
author: Manali Dangarikar
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import pickle as pkl
import pandas as pd
import os
import csv
import json
from sample import Sample



def loadmnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.cast(x_train / 255.0, tf.float32)
    x_test = tf.cast(x_test / 255.0, tf.float32)
    
    return (x_train, y_train), (x_test, y_test)

def saveModel(object, filepath):
    """
    dump the model to a location in pickle format

    :param object: som configuration
    :type object: dict
    :param filepath: save location
    :type filepath: str
    """
    pkl.dump(object, open(filepath, 'wb'))

def loadModel(filepath):
    """
    load the model from a location in pickle format

    :param filepath: save location
    :type filepath: str
    """
    object = pkl.load(open(filepath, 'rb'))
    return object

def dumpjson(data, filepath):
    with open(filepath, 'w') as fp:
        json.dump(data, fp)

def loadjson(filepath):
    data = None
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data

def loadSplitData(path, class_number):
    """
    Load class specific data

    :param path: file path
    :type path: str
    :param class_number: current class number
    :type class_number: int
    :return: samples
    :rtype: numpy array of Sample() objects
    """
    samples = pkl.load(open(os.path.join(path, str(class_number) + ".pkl"), 'rb'))
    return samples

def loadClassIncremental(path, taskNumber, taskSize):
    samples = np.array([])
    for t in range(taskNumber*taskSize, (taskNumber+1)*taskSize):
        task_samples = loadSplitData(path, t)
        samples = np.concatenate([samples, task_samples])
    np.random.shuffle(samples)
    return samples

def writeAccuracy(path, accuracy):
    """
    Write the accuracy score for the model to a file

    :param path: filepath
    :type path: str
    :param accuracy: accuracy value
    :type accuracy: float
    """
    with open(path, 'w') as fp:
        fp.write("accuracy: " + str(accuracy))

def generateSamples(images, labels, shapeX, shapeY):
    """
    Generate a numpy array of Sample() objects from give images and labels

    :param images: images
    :type images: numpy 3D array
    :param labels: labels
    :type labels: numpy 1D array
    :param shapeX: shapeX of image
    :type shapeX: int
    :param shapeY: shapeY of image
    :type shapeY: int
    :return: samples
    :rtype: numpy 1D array
    """
    samples = []
    for image, label in zip(images, labels):
        samples.append(Sample(label, shapeX, shapeY, image))
    samples = np.asarray(samples)
    return samples

def loadNistTestData(path):
    """
    For either of MNIST, FashionMNIST, KMNIST:
    load test images and labels from their idx files
    then form Sample() object for all (image, label)
    return numpy array of all such Sample() objects

    :param path: path to idx file of test data
    :type path: str
    :return: samples
    :rtype: numpy array of Sample()
    """
    # test_images is a 3D numpy array
    test_images = idx2numpy.convert_from_file(os.path.join(path, 't10k-images-idx3-ubyte')) / 255.0
    # test_labels is 1D numpy array
    test_labels = idx2numpy.convert_from_file(os.path.join(path, 't10k-labels-idx1-ubyte')) 

    samples = generateSamples(test_images, 
                                    test_labels,
                                    test_images.shape[1], 
                                    test_images.shape[2]
                                    )

    return samples


def loadNistTrainData(path):
    """
    For either of MNIST, FashionMNIST, KMNIST:
    load test images and labels from their idx files
    then form Sample() object for all (image, label)
    return numpy array of all such Sample() objects

    :param path: path to idx file of test data
    :type path: str
    :return: samples
    :rtype: numpy array of Sample()
    """
    # test_images is a 3D numpy array
    train_images = idx2numpy.convert_from_file(os.path.join(path, 'train-images-idx3-ubyte')) / 255.0
    # test_labels is 1D numpy array
    train_labels = idx2numpy.convert_from_file(os.path.join(path, 'train-labels-idx1-ubyte')) 

    samples = generateSamples(train_images, 
                                    train_labels,
                                    train_images.shape[1], 
                                    train_images.shape[2]
                                    )

    return samples

def loadCifarTestData(path):
    """
    load test dataset cifar-10
    then form Sample() object for all (image, label)
    return numpy array of all such Sample() objects

    :param path: file path
    :type path: str
    :return: samples
    :rtype: numpy array of Sample()
    """
    with open(path, 'rb') as fo:
        dict = pkl.load(fo, encoding='bytes')
    test_images = dict[b'data']

    # test_images is a 3D numpy array
    test_images = np.reshape(test_images, [-1, 32, 32])
    # test_labels is 1D numpy array
    test_labels = np.asarray(dict[b'labels'])

    samples = generateSamples(test_images, 
                                    test_labels,
                                    test_images.shape[1], 
                                    test_images.shape[2]
                                    )

    return samples

def loadDomainIncremental(path, taskNumber, taskSize):
    samples = np.array([])
    for idx,t in enumerate(range(taskNumber*taskSize, (taskNumber+1)*taskSize)):
        task_samples = loadSplitData(path, t)
        for s in range(task_samples.shape[0]):
            task_samples[s].setLabel(idx)
        samples = np.concatenate([samples, task_samples])
    np.random.shuffle(samples)
    return samples
    # samples = self.loadNistTrainData(path)
    # task_samples = []
    # for sample in samples:
    #     label = sample.getLabel()
    #     if ((taskNumber * taskSize) <= label < (taskNumber+1) * taskSize):
    #         sample.setLabel(label - (taskNumber * taskSize))
    #         task_samples.append(sample)
    # task_samples = np.asarray(task_samples)
    # np.random.shuffle(task_samples)
    # return task_samples

def dumpSplitData(images, labels, nClasses, path):
    """
    class wise split the data and dump pickle files

    :param images: images
    :type images: numpy 3D array
    :param labels: image labels
    :type labels: numpy 1D array
    :param nClasses: number of classes
    :type nClasses: int
    :param path: directory path
    :type path: str
    """
    for c in range(nClasses):
        indexes = np.where(labels == c)[0]
        class_images = images[indexes]
        class_labels = labels[indexes]
        samples = generateSamples(class_images, class_labels, images.shape[1], images.shape[-1])
        pkl.dump(samples, open(os.path.join(path, str(c) + ".pkl"), 'wb'))
    
