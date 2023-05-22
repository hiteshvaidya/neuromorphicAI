"""
This file contains code for performing grid search on SOM to find out optimum set of hyperparameters that produce the best accuracy results

version: 1.0
author: Hitesh Vaidya
"""

# import libraries
from sklearn.model_selection import GridSearchCV
from network import Network
from sklearn.model_selection import ParameterGrid
import argparse
import dataloader
import tensorflow as tf
import os
from testSOM import testClass

def create_class_incremental_model(units, radius, learning_rate, variance, variance_alpha, tau_radius, tau_lr):
    """
    create an SOM model

    :param units: number of units in one row/column
    :type units: int
    :param radius: initial radius
    :type radius: float
    :param learning_rate: initial learning rate
    :type learning_rate: float
    :param variance: initial running variance
    :type variance: float
    :param variance_alpha: alpha value for running variance
    :type variance_alpha: float
    :param tau_radius: tau constant for radius decay
    :type tau_radius: float
    :param tau_lr: tau constant for learning rate decay
    :type tau_lr: float
    :return: SOM model
    :rtype: Network() object
    """
    model = Network(28, units, 10, radius, learning_rate, variance, variance_alpha, tau_radius, tau_lr)

    return model

if __name__ == '__main__':
    
    # define parser for command line arguments
    parser = argparse.ArgumentParser()

    # add command line arguments
    parser.add_argument('-dc', '--data_choice', type=str, default=None, help='mnist, fashion, kmnist')

    args = parser.parse_args()

    # Define the hyperparameter search space
    param_grid = {
        'u': [15, 20, 25, 30],           # number of units
        'v': [0.4, 0.5],                 # initial running variance
        'va': [0.9, 0.95, 0.99],         # running variance alpha
        'r': [1.5, 2],                   # initial radius
        'lr': [0.05, 0.06, 0.07, 0.08],  # initial learning rate
        'd': [args.data_choice],         # dataset choice
        'tr': [2, 4, 5, 8, 10, 12, 15],  # tau value for radius
        'tlr': [30, 35, 40, 45, 50]     # tau for learning rate     
    }

    if not os.path.exists(os.path.join("logs", 'class_incremental')):
        os.makedirs(os.path.join("logs", 'class_incremental'))
    
    if not os.path.exists(os.path.join("logs", 
                                       'class_incremental', 
                                       args.data_choice)):
        os.makedirs(os.path.join("logs", 
                                 'class_incremental', 
                                 args.data_choice))
    

    # generate all possible combinations of hyperparameters
    grid = ParameterGrid(param_grid)

    # Perform grid search
    best_accuracy = 0.0
    best_model = None
    best_model_path = os.path.join('logs', 'class_incremental', args.data_choice, 'best_model')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    # load test samples
    test_samples = dataloader.loadNistTestData("../data/" + args.data_choice)

    for trial, params in enumerate(grid):
        print("trial: ", trial)
        # path for log folder
        output_path = os.path.join(os.getcwd(), 'logs', 'class_incremental', params['d'], 'grid_trial-' + str(trial))

        # create the model with current hyperparameters
        model = create_class_incremental_model(params['u'], params['r'], params['lr'], params['v'], params['va'], params['tr'], params['tlr'])
        
        for task_index in range(5):
            # Load data samples for each task
            train_samples = dataloader.loadClassIncremental(os.path.join('../data', params['d'], 'train'), task_index, 2)

            # train model
            model.fit(train_samples, output_path, task_index)

        # save the model
        config = model.getConfig()
        
        # free some memory in gpu by deleting trained model
        # dataloader.saveModel(config, os.path.join('logs', 'class_incremental', args.data_choice, 'temp_model'))
        del model

        # declare test model
        test_model = testClass(config['som'], 
                    config['shapeX'], 
                    config['shapeY'], 
                    config['unitsX'], 
                    config['unitsY'], 
                    config['class_count'], 
                    10)
        # set the label for every unit in test model
        test_model.setPMILabel()
        
        predictions = []
        labels = []
        for sample in (test_samples): 
            feature_map = test_model.layer1(config['som'], 
                                            sample.getImage())
            bmu = test_model.layer2(feature_map)
            output = test_model.getPredictedClass(bmu)
            predictions.append(output)
            labels.append(sample.getLabel())
        predictions = tf.cast(tf.stack(predictions), dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.float32)
        
        accuracy = test_model.getAccuracy(predictions, labels) * 100

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            dataloader.saveModel(config, os.path.join(best_model_path, 'model_config.pkl'))
            dataloader.writeAccuracy(os.path.join(best_model_path, 'accuracy.txt'), best_accuracy)
            best_model_output_path = output_path


    # Print the best hyperparameters
    print("Best accuracy: ", best_accuracy)
    print("Best Hyperparameters: ", best_params)
    # write accuracy and hyperparameters to the best model log folder
    dataloader.writeAccuracy(os.path.join(best_model_path, 'accuracy.txt'))
    config = dataloader.loadModel(os.path.join(best_model_path, 'model_config.pkl'))
    config['params'] = best_params
    dataloader.saveModel(config, os.path.join(best_model_path, 'model_config.pkl'))