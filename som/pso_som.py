# import libraries
from sklearn.model_selection import GridSearchCV
from network import Network
from sklearn.model_selection import ParameterGrid
import argparse
import dataloader
import tensorflow as tf
import os
from testSOM import testClass
from pyswarm import pso
import util

counter = 0
dataset_choice = 'mnist'

def objective_function(model_config):
    # test SOM

    # load test samples
    test_samples = dataloader.loadNistTestData("../data/" + dataset_choice)
    
    # declare test model
    test_model = testClass(model_config[0], model_config[1], model_config[2], 
                           model_config[3], model_config[4], model_config[5], model_config[6])
    # set the label for every unit in test model
    test_model.setPMILabel()

    predictions = []
    labels = []
    for sample in (test_samples): 
        feature_map = test_model.layer1(model_config[0], 
                                        sample.getImage())
        bmu = test_model.layer2(feature_map)
        output = test_model.getPredictedClass(bmu)
        predictions.append(output)
        labels.append(sample.getLabel())
    predictions = tf.cast(tf.stack(predictions), dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.float32)
    
    accuracy = util.getAccuracy(predictions, labels) * 100
    
    return accuracy

def fitness_function(hyperparameters):
    # train SOM
    global counter
    counter += 1

    print('trial - ', str(counter))
    # create the model with current hyperparameters
    model = create_class_incremental_model(hyperparameters)
    
    for task_index in range(5):
        # Load data samples for each task
        train_samples = dataloader.loadClassIncremental(os.path.join('../data', dataset_choice, 'train'), task_index, 2)

        output_path = os.path.join(os.getcwd(), 'logs', 'class_incremental', dataset_choice, 'pso-trial-' + str(counter))
        # train model
        model.fit(train_samples, output_path, task_index)
    
    config = model.getConfig()
    del model
    parameters = [config['som'], 
                config['shapeX'], 
                config['shapeY'], 
                config['unitsX'], 
                config['unitsY'], 
                config['class_count'], 
                10]
    
    del config
    accuracy = objective_function(parameters)

    return accuracy


# def create_class_incremental_model(units, radius, learning_rate, variance, variance_alpha, tau_radius, tau_lr):
def create_class_incremental_model(hyperparameters):
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
    # hyperparameters - [units, radius, learning_rate, running_variance, running_variance_alpha, tau_radius, tau_learning_rate]
    model = Network(28, hyperparameters[0], 
                    10, hyperparameters[1], 
                    hyperparameters[2], 
                    hyperparameters[3],
                    hyperparameters[4],
                    hyperparameters[5],
                    hyperparameters[6])

    return model

if __name__ == '__main__':

    # Define the hyperparameter search space
    # bounds - [units, radius, learning_rate, running_variance, running_variance_alpha, tau_radius, tau_learning_rate]
    lower_bounds = [15, 1.5, 0.05, 0.4, 0.9, 2, 30]
    upper_bounds = [30, 2, 0.08, 0.5, 0.99, 15, 50]

    if not os.path.exists(os.path.join("logs", 'class_incremental')):
        os.makedirs(os.path.join("logs", 'class_incremental'))
    
    if not os.path.exists(os.path.join("logs", 
                                       'class_incremental', 
                                       dataset_choice)):
        os.makedirs(os.path.join("logs", 
                                 'class_incremental', 
                                 dataset_choice))
   
    # Perform grid search
    best_accuracy = 0.0
    best_model_path = os.path.join('logs', 'class_incremental', dataset_choice, 'pso_best_model')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    # Use PSO to find the best hyperparameters
    hyperparameters_opt, fitness_opt = pso(fitness_function, lower_bounds, upper_bounds)

    print("Best hyperparameters:", hyperparameters_opt)
    print("Best fitness value:", fitness_opt)

    # write accuracy and hyperparameters to the best model log folder
    dataloader.writeAccuracy(os.path.join(best_model_path, 'accuracy.txt'))
    with open(os.path.join(best_model_path, 'best_hyperparameters.txt'), 'wb') as fp:
        fp.write(', '.join(str(elem) for elem in hyperparameters_opt))