"""
network.py

description: code for DendSOM
version: 2.0
author: Hitesh Vaidya
"""

# import libraries
import tensorflow as tf
import os
from layer import CustomDistanceLayer
from CosineDistanceLayer import CosineDistanceLayer
from sample import Sample
import dataloader
from MinLayer import MinLayer
from MaxLayer import MaxLayer
import util
import cv2
import math
import time
import numpy as np
import argparse
from testSOM import testClass
from tqdm import tqdm
import json
import multiprocessing
import pickle as pkl
from datetime import datetime
import concurrent.futures

class Network(tf.keras.Model):
    """
    This class forms the network for the operation of SOM

    :param tf: tensorflow
    :type tf: tensorflow layer object
    """
    def __init__(self, imgSize, n_units, n_classes, radius, learning_rate, 
                r_exp, alpha_crit):
        """
        constructor

        :param imgSize: number of pixels in a row and column of the input image or a unit in the SOM
        :type imgSize: int
        :param n_units: number of units in the SOM
        :type n_units: int
        """
        super(Network, self).__init__()
        st = time.time()
        # Total number of pixels in a row and column of SOM
        self.shapeX = imgSize * n_units
        self.shapeY = imgSize * n_units

        # Total number of units in the SOM
        self.unitsX = n_units
        self.unitsY = n_units

        # randomly initialize every unit (n_units * n_units) in the SOM
        units = []
        for _ in range(self.unitsX * self.unitsY):
            current_time = time.time() - st
            units.append(tf.random.normal([imgSize, imgSize], mean=0.4, stddev=0.3, seed=current_time))
        
        # stack the units along rows and columns or reshape to form the SOM
        self.som = tf.concat([tf.concat(units_col, axis=1) for units_col in tf.split(units, self.unitsX)], axis=0)
        # reshape the SOM
        self.som = tf.reshape(self.som, [self.shapeY, self.shapeX])

        # clip values between 0 and 1
        self.som = tf.clip_by_value(self.som, 0.0, 1.0)

        # class_count for every unit i.e. how many number of times a unit was selected as BMU for every class in the dataset
        self.class_count = tf.zeros([n_units, n_units, n_classes])

        # Calculate distances between units of the SOM
        self.cartesian_distances = util.calculate_distances(n_units, n_units)
        
        # Create a matrix for radius of every unit
        self.initial_radius = radius
        self.radius = radius

        # Create a matrix for learning rate of every unit
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
    

        # t iterations
        self.t = 0
        self.r_exp = r_exp
        self.alpha_crit = alpha_crit
        self.iter_crit = tf.math.log(tf.math.floor(1E3 *(learning_rate / alpha_crit)) )
        
        # Declare the layers of the network
        # self.layer1 = CustomDistanceLayer(imgSize, n_units)
        self.layer1 = CosineDistanceLayer(imgSize, n_units)
        self.layer2 = MaxLayer(n_units)

    def decayRadius(self):
        """
        Decay current radius value of the best matching unit

        :param bmu: best matching unit
        :type bmu: tuple
        """
        self.radius = self.initial_radius * tf.exp(-self.t / 1E3)
        # self.radius = decay * tf.ones([self.unitsX, self.unitsY])
       

    def decayLearningRate(self):
        """
        Decay current learning rate of the best matching unit

        :param bmu: best matching unit
        :type bmu: tuple
        """
        self.learning_rate = self.initial_learning_rate * tf.exp(-self.t / 1E3)
        # self.learning_rates = decay * tf.ones([self.unitsX, self.unitsY])
        
        
    def visualize_model(self):
        """
        Visualize the current state of SOM
        """
        # Display current state of SOM
        cv2.imshow("Self Organizing Map", self.som.numpy())
        # wait for 1 milli-seconds
        cv2.waitKey(1)
    
    def displayVariance(self):
        """
        Display the running variance values of all SOM units
        """
        cv2.imshow("SOM variance", self.running_variance.numpy())
        cv2.waitKey(1)
    
    def saveImage(self, folder_path, index):
        """
        save an image of the current state of SOM

        :param folder_path: folder location
        :param folder_path: str
        :param index: current class index
        :type index: int
        """
        image = self.som.numpy()
        image = (image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(folder_path, str(index) + ".png"), image)
    
    def resetIterationIndex(self):
        if self.t % self.iter_crit == 0:
            self.t = math.floor(self.t / self.r_exp)

    def weight_update(self, bmu, input_matrix, label):
        # Decay parameters
        self.decayRadius()
        self.decayLearningRate()

        # create a distance modifier for neighbourhood function
        distance = self.cartesian_distances[:, :, bmu[0], bmu[1]]

        final_modifier = self.learning_rate * tf.math.exp((-1.0 * distance * distance) / (2 * self.radius))

        final_modifier = tf.repeat(final_modifier, repeats=self.shapeY // self.unitsY, axis=1)
        final_modifier = tf.repeat(final_modifier, repeats=self.shapeX // self.unitsX, axis=0)
        final_modifier = tf.reshape(final_modifier, [self.shapeX, self.shapeY])

        input_matrix = tf.repeat(input_matrix, repeats=self.unitsY, axis=1)
        input_matrix = tf.repeat(input_matrix, repeats=self.unitsX, axis=0)
        # Perform the weight update
        self.som += final_modifier * (input_matrix - self.som)

        # clip the values of SOM
        self.som = tf.clip_by_value(self.som, 0.0, 1.0)

        self.class_count = tf.tensor_scatter_nd_update(self.class_count, [[bmu[0], bmu[1], label]], [self.class_count[bmu[0], bmu[1], label] + 1])  
    
    def call(self, x, y):
        """
        Forward pass through the network

        :param x: input image
        :type x: matrix of float values
        """
        unit_map = self.layer1(self.som, x)
        bmu = self.layer2(unit_map)
        self.weight_update(bmu, x, y)
    
    def fit(self, sample, folder_path, index, task_size, training_type):
        """
        Train the model

        :param train_samples: samples from train set
        :type train_samples: array of sample objects
        :param folder_path: folder path for storing images
        :type folder_path: str
        :param index: current task index
        :type index: int
        """
        # tqdm.write("fitting model for task " + str(index))
        # for cursor, sample in tqdm(enumerate(train_samples)):
            # forward pass
        # print(train_samples[0].getImage().shape)
        label = util.getTrainingLabel(sample.getLabel(),
                                        task_size,
                                        training_type)
        self(sample.getImage(), label)
            
        # save the image of current state of SOM
        self.saveImage(folder_path, index)


    def getConfig(self):
        """
        Get configuration of this model for saving it

        :return: configuration of the network class
        :rtype: dict
        """
        config = {}
        config['som'] = self.som
        config['shapeX'] = self.shapeX
        config['shapeY'] = self.shapeY
        config['unitsX'] = self.unitsX
        config['unitsY'] = self.unitsY
        config['class_count'] = self.class_count
        config['radius'] = self.radius
        config['learning_rate'] = self.learning_rate

        return config


def fit_model(model, samples, folder_path, index, task_size, training_type, timeStep):
    model.fit(samples, folder_path, index, task_size, training_type, timeStep)

if __name__ == '__main__':

    # define parser for command line arguments
    parser = argparse.ArgumentParser()

    # add command line arguments
    parser.add_argument('-u', '--units', type=int, required=True, default=10, help='number of units in a row of the SOM')
    parser.add_argument('-r', '--radius', type=float, required=True, default=None, help='initial radius of every unit in SOM')
    parser.add_argument('-lr', '--learning_rate', type=float, required=True, default=None, help='initial learning rate of every unit in SOM')
    parser.add_argument('-ac', '--alpha_crit', type=float, required=False, default=0.9, help='initial value of alpha for running variance')
    parser.add_argument('-re', '--r_exp', type=float, required=False, default=0.5, help='initial value of running variance')
    parser.add_argument('-fp', '--filepath', type=str, required=False, default=None, help='filepath for saving trained SOM model')
    parser.add_argument('-d', '--dataset', type=str, default=None, help='dataset type mnist/fashion/kmnist/cifar')
    parser.add_argument('-nt', '--n_tasks', type=int, default=10, help='number of tasks in incremental training')
    parser.add_argument('-ts', '--task_size', type=int, default=1, help='number of classes per task in incremental training')
    parser.add_argument('-t', '--training_type', type=str, default='class', help='class incremental or domain incremental training')
    parser.add_argument('-p', '--patch_size', type=int, required=True, default=None, help='size of each patch of an image')
    parser.add_argument('-s','--stride', type=int, default=None, required=True, help='stride length for breaking image into patches')
    args = parser.parse_args()
    
    # start time
    begin = datetime.now()

    # create 'logs' folder
    if not os.path.exists("logs"):
        os.makedirs("logs")
    # create a folder for current experiment
    folder_path = os.path.join(os.getcwd(), 'logs', args.filepath)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("folder_path: ", folder_path)

    # Declare a list of objects of SOM where each SOM runs on a split patch of an input sample
    networks = []
    network_ids = []
    # number of classes in complete training
    n_classes = 0
    if args.training_type == 'class':
        n_classes = args.n_tasks * args.task_size
    elif args.training_type == 'domain':
        n_classes = args.task_size
    
    n_soms = (1 + (28 - args.patch_size)//args.stride) * (1 + (28 - args.patch_size)//args.stride)
    print('nsoms: ', n_soms)

    for count in range(n_soms):
        networks.append(Network(args.patch_size, args.units, n_classes, 
                                args.radius, args.learning_rate, 
                                args.r_exp, args.alpha_crit
                                )
                        )
        network_ids.append(count)

    b = util.dendSOMTaskAccuracy(networks, 
                             args.dataset, 
                             args.patch_size, 
                             args.stride,
                             args.n_tasks,
                             args.task_size,
                             n_soms,
                             args.training_type)
    
    print('b = ', b)


    final_accuracies = tf.constant([], dtype=tf.float32, 
                                   shape=(0, args.n_tasks))
    num_threads = n_soms
    
    tqdm.write('training on tasks')
    # Perform the forward pass
    for index in tqdm(range(args.n_tasks)):
        # Load the data as per choice of training
        train_samples = None
        if args.training_type == 'class':
            train_samples = dataloader.loadClassIncremental(
                os.path.join("../data", args.dataset, "train"), 
                index, args.task_size)
        elif args.training_type == 'domain':
            train_samples = dataloader.loadDomainIncremental(
                os.path.join("../data", args.dataset, "train"), 
                index, args.task_size)
        
        # split every image into patches
        train_samples = dataloader.breakImages(train_samples,
                                              args.patch_size,
                                              args.stride)
   
        # fit/train the model on train samples
        for sample in train_samples:
            print('sample: ', sample.shape)
            for count in range(n_soms):
                networks[count].fit(sample[count], 
                                    folder_path, 
                                    index,
                                    args.task_size,
                                    args.training_type)
                networks[count].resetIterationIndex()

        task_accuracy = util.dendSOMTaskAccuracy(networks, 
                                                 args.dataset, 
                                                 args.patch_size,
                                                 args.stride, 
                                                 args.n_tasks,
                                                 args.task_size, 
                                                 n_soms,
                                                 args.training_type)
        final_accuracies = tf.concat([final_accuracies, 
                                tf.reshape(task_accuracy, [1, -1])], 
                                axis=0)

    for count in range(n_soms):
        config = networks[count].getConfig()
        pkl.dump(config, 
                open(os.path.join(folder_path, 
                                  'model_config-' + str(count) + '.pkl'
                                  ), 
                    'wb')
                )
    del networks
    
    fwt = util.getFWT(final_accuracies, b)
    bwt = util.getBWT(final_accuracies)
    avgAccuracy = util.getAverageAccuracy(final_accuracies)
    learningAccuracy = util.getLearningAccuracy(final_accuracies)
    forgettingMeasure = util.getForgettingMeasure(final_accuracies)

    print('b = ', b)
    print('fwt: ', fwt)
    print('bwt: ', bwt)
    print('average accuracy: ', avgAccuracy)
    print('learning accuracy: ', learningAccuracy)
    print('forgetting measure: ', forgettingMeasure)
        
    output_path = os.path.join(folder_path, 'transfer_metrics.csv')
    
    numpy_array = final_accuracies.numpy()
    np.savetxt(output_path, numpy_array, delimiter=', ', fmt='%.4f')

    with open(os.path.join(folder_path, 'tranfer_metrics.txt'), 'w') as fp:
        fp.write('b = ' + ", ".join([str(x) for x in b]) + '\n')
        fp.write('fwt = ' + str(fwt) + '\n')
        fp.write('bwt = ' + str(bwt) + '\n')
        fp.write('average accuracy = ' + str(avgAccuracy) + '\n')
        fp.write('learning accuracy = ' + str(learningAccuracy) + '\n')
        fp.write('forgetting measure: ' + str(forgettingMeasure))

    end = datetime.now()
    
    print("execution time: ", (end-begin).total_seconds() / 60)