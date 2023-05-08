"""
dataloader.py

description: This file contains code for dataloader
version: 1.0
author: Manali Dangarikar
"""
import tensorflow as tf
import pickle as pkl
import os
from sample import Sample
import numpy as np


def loadmnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.cast(x_train / 255.0, tf.float32)
    x_test = tf.cast(x_test / 255.0, tf.float32)
    
    return (x_train, y_train), (x_test, y_test)

def saveModel(model, filepath):
    """
    dump the model to a location in pickle format

    :param model: SOM network
    :type model: Network() object
    :param filepath: save location
    :type filepath: str
    """
    pkl.dump(model, open(filepath, 'wb'))

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
