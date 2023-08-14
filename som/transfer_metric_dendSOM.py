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
from sample import Sample
import dataloader
from MinLayer import MinLayer
import util
import cv2
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
                running_variance, running_variance_alpha, tau_radius, tau_lr):
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

        # Initialize the matrix for running variance
        self.running_variance = tf.ones([self.shapeX, self.shapeY]) * running_variance
        # alpha value for updating running variance
        self.running_variance_alpha = running_variance_alpha

        # Calculate distances between units of the SOM
        self.cartesian_distances = util.calculate_distances(n_units, n_units)
        
        # Create a matrix for radius of every unit
        self.initial_radius = radius
        self.radius = radius * tf.ones([n_units, n_units])

        # Create a matrix for learning rate of every unit
        self.initial_learning_rates = learning_rate
        self.learning_rates = learning_rate * tf.ones([n_units, n_units])
        
        # set the tau constants for parameter decay
        self.tau_radius = tau_radius
        self.tau_lr = tau_lr
        
        # Declare the layers of the network
        self.layer1 = CustomDistanceLayer(imgSize, n_units)
        self.layer2 = MinLayer(n_units)

    def decayRadius(self, bmu):
        """
        Decay current radius value of the best matching unit

        :param bmu: best matching unit
        :type bmu: tuple
        """
        # 15 is a constant value (can be changed)
        decay = self.initial_radius * tf.exp(-tf.reduce_sum(self.class_count[bmu[0], bmu[1], :]) / self.tau_radius)
        self.radius = tf.tensor_scatter_nd_update(self.radius, [bmu], [decay])
        self.radius = tf.math.maximum(0.00001 * tf.ones(
                                        tf.shape(self.radius)), 
                                        self.radius)

    def decayLearningRate(self, bmu):
        """
        Decay current learning rate of the best matching unit

        :param bmu: best matching unit
        :type bmu: tuple
        """
        # 25 is a constant value (can be changed)
        decay = self.initial_learning_rates * tf.exp(-tf.reduce_sum(self.class_count[bmu[0], bmu[1], :]) / self.tau_lr)
        self.learning_rates = tf.tensor_scatter_nd_update(self.learning_rates,
                                                    [bmu], [decay])
        self.learning_rates = tf.math.maximum(0.00001 * tf.ones(
                                        tf.shape(self.learning_rates)), 
                                        self.learning_rates)
        
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
        variance = self.running_variance.numpy()
        image = (image * 255).astype(np.uint8)
        variance = (variance * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(folder_path, str(index) + ".png"), image)
        cv2.imwrite(os.path.join(folder_path, str(index) + "_variance.png"), variance)

    def weight_update(self, bmu, input_matrix, label):
        # create a distance modifier for neighbourhood function
        distance_modifier = 1.0 / (2.0 * self.radius[bmu[0], bmu[1]] * self.radius[bmu[0], bmu[1]])

        # create a constant used for updating variance_alpha
        constant = -1.0 * tf.math.log(1E-8 / self.learning_rates[bmu[0], bmu[1]]) / distance_modifier

        # We do not perform weight update for units that are farther in the neighbourhood of BMU, than the radius of BMU 
        # This mean, we use the radius of BMU as threashold and set distance values of all the units farther than the threshold to be 0 so as to avoid weight update for them
        mask = tf.where(self.cartesian_distances[ :, :, bmu[0], bmu[1]] > self.radius[bmu[0], bmu[1]], tf.zeros_like(self.cartesian_distances[ :, :, bmu[0], bmu[1]]), tf.ones([self.unitsX, self.unitsY], dtype=tf.float32))
        

        final_modifier = mask * self.learning_rates * tf.math.exp(-self.cartesian_distances[:, :, bmu[0], bmu[1]] * distance_modifier)

        final_modifier = tf.repeat(final_modifier, repeats=self.shapeY // self.unitsY, axis=1)
        final_modifier = tf.repeat(final_modifier, repeats=self.shapeX // self.unitsX, axis=0)
        final_modifier = tf.reshape(final_modifier, [self.shapeX, self.shapeY])

        # Perform the weight update
        self.som += final_modifier * (input_matrix - self.som)

        # clip the values of SOM
        self.som = tf.clip_by_value(self.som, 0.0, 1.0)

        self.class_count = tf.tensor_scatter_nd_update(self.class_count, [[bmu[0], bmu[1], label]], [self.class_count[bmu[0], bmu[1], label] + 1])

        # we need some alpha value to update running variance 
        variance_alpha  = (self.running_variance_alpha - 0.5) + 1.0 / (1.0 + tf.math.exp(-self.cartesian_distances[:, :, bmu[0], bmu[1]] / constant))
        
        # we only update variance of units that are within the range of the radius of our BMU. We obtain this mask from 'modifier' variable
        # For all the units farther than the radius of BMU, alpha value will be 1.0 because we do not want to update their variance at line 158
        variance_alpha = variance_alpha * mask + (1 - mask)
        variance_alpha = tf.clip_by_value(variance_alpha, 0.0, 1.0)
        variance_alpha = tf.repeat(variance_alpha, repeats=self.shapeX // self.unitsX, axis=1)
        variance_alpha = tf.repeat(variance_alpha, repeats=self.shapeY // self.unitsY, axis=0)
        variance_alpha = tf.reshape(variance_alpha, [self.shapeX, self.shapeY])
        
        # variance_alpha = tf.tile(variance_alpha, [self.shapeX // self.unitsX, self.shapeY // self.unitsY])
        
        # Update running variance of SOM
        self.running_variance = variance_alpha * self.running_variance + (1.0 - variance_alpha) * (input_matrix - self.som) * (input_matrix - self.som)

        # Decay parameters
        self.decayRadius(bmu)
        self.decayLearningRate(bmu)
    
    def call(self, x, y):
        """
        Forward pass through the network

        :param x: input image
        :type x: matrix of float values
        """
        tiled_input, unit_map = self.layer1(self.som, self.running_variance, x)
        bmu = self.layer2(unit_map)
        self.weight_update(bmu, tiled_input, y)
    
    def fit(self, train_samples, folder_path, index, task_size, training_type):
        """
        Train the model

        :param train_samples: samples from train set
        :type train_samples: array of sample objects
        :param folder_path: folder path for storing images
        :type folder_path: str
        :param index: current task index
        :type index: int
        """
        tqdm.write("fitting model for task " + str(index))
        for cursor, sample in tqdm(enumerate(train_samples)):
            # forward pass
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
        config['running_variance'] = self.running_variance
        config['radius'] = self.radius
        config['learning_rates'] = self.learning_rates
        config['tau_radius'] = self.tau_radius
        config['tau_lr'] = self.tau_lr

        return config


def fit_model(model, samples, folder_path, index, task_size, training_type):
    model.fit(samples, folder_path, index, task_size, training_type)

if __name__ == '__main__':

    # define parser for command line arguments
    parser = argparse.ArgumentParser()

    # add command line arguments
    parser.add_argument('-g', '--gpuid', type=str, default=None, help='gpu id')
    parser.add_argument('-u', '--units', type=int, required=True, default=10, help='number of units in a row of the SOM')
    parser.add_argument('-r', '--radius', type=float, required=True, default=None, help='initial radius of every unit in SOM')
    parser.add_argument('-lr', '--learning_rate', type=float, required=True, default=None, help='initial learning rate of every unit in SOM')
    parser.add_argument('-va', '--variance_alpha', type=float, required=False, default=0.9, help='initial value of alpha for running variance')
    parser.add_argument('-v', '--variance', type=float, required=False, default=0.5, help='initial value of running variance')
    parser.add_argument('-fp', '--filepath', type=str, required=False, default=None, help='filepath for saving trained SOM model')
    parser.add_argument('-d', '--dataset', type=str, default=None, help='dataset type mnist/fashion/kmnist/cifar')
    parser.add_argument('-tr', '--tau_radius', type=float, default=None, help='tau constant for decaying radius')
    parser.add_argument('-tlr', '--tau_lr', type=float, default=None, help='tau constant for decaying learning rate')
    parser.add_argument('-n', '--n_soms', type=int, required=True, default=None, help='number of SOMs')
    parser.add_argument('-nt', '--n_tasks', type=int, default=10, help='number of tasks in incremental training')
    parser.add_argument('-ts', '--task_size', type=int, default=1, help='number of classes per task in incremental training')
    parser.add_argument('-t', '--training_type', type=str, default='class', help='class incremental or domain incremental training')
    parser.add_argument('-us', '--unit_size', type=int, required=True, default=None, help='size of each patch of an image')
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

    for count in range(args.n_soms):
        networks.append(Network(args.unit_size, args.units, n_classes, 
                                args.radius, args.learning_rate, 
                                args.variance, args.variance_alpha, 
                                args.tau_radius, args.tau_lr
                                )
                        )
        network_ids.append(count)

    b = util.dendSOMTaskAccuracy(networks, 
                             args.dataset, 
                             args.unit_size, 
                             args.n_tasks,
                             args.task_size,
                             args.n_soms,
                             args.training_type)
    
    print('b = ', b)


    final_accuracies = tf.constant([], dtype=tf.float32, shape=(0, n_classes))
    num_threads = args.n_soms

    # Perform the forward pass
    for index in range(args.n_tasks):
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
                                              args.unit_size)
        
        # fit/train the model on train samples
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit the 'fit_model' function for each model in the list
            futures = [executor.submit(fit_model, 
                                       networks[count], 
                                       train_samples[:, count],
                                       folder_path,
                                       index,
                                       args.task_size,
                                       args.training_type) 
                                       for count in range(args.n_soms)]

            # Wait for all futures (fitting processes) to complete
            concurrent.futures.wait(futures)

        # fit/train the model on train samples
        # for count in range(args.n_soms):
        #     networks[count].fit(train_samples[:, count], 
        #                         folder_path, 
        #                         index,
        #                         args.task_size,
        #                         args.training_type)

        task_accuracy = util.dendSOMTaskAccuracy(networks, 
                                                 args.dataset, 
                                                 args.unit_size, 
                                                 args.n_tasks,
                                                 args.task_size, 
                                                 args.n_soms,
                                                 args.training_type)
        final_accuracies = tf.concat([final_accuracies, 
                                tf.reshape(task_accuracy, [1, -1])], 
                                axis=0)

    for count in range(args.n_soms):
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