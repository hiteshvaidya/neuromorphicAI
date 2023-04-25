"""
dataloader.py

description: This file contains code for dataloader
version: 1.0
author: Manali Dangarikar
"""
import tensorflow as tf

def loadmnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.cast(x_train / 255.0, tf.float32)
    x_test = tf.cast(x_test / 255.0, tf.float32)
    
    return (x_train, y_train), (x_test, y_test)