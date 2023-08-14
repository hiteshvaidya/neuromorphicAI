from network import Network
import argparse
import dataloader
import os
import tensorflow as tf
import util
from testSOM import testClass
import numpy as np
import pickle as pkl

def create_class_incremental_model(unit_size, units, n_classes, radius, learning_rate, variance, variance_alpha, tau_radius, tau_lr):
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
    model = Network(unit_size, units, n_classes, radius, learning_rate, variance, variance_alpha, tau_radius, tau_lr)

    return model

def get_accuracies(predictions, labels):
    accuracies = list(range(len(predictions)))

    for index in range(len(accuracies)):
        accuracies[index] = predictions[index] / labels[index]

    return accuracies


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
    parser.add_argument('-nt', '--n_tasks', type=int, default=10, help='number of tasks in incremental training')
    parser.add_argument('-ts', '--task_size', type=int, default=1, help='number of classes per task in incremental training')
    parser.add_argument('-t', '--training_type', type=str, default='class', help='class incremental or domain incremental training')
    parser.add_argument('-us', '--unit_size', type=int, required=True, default=None, help='size of each patch of an image')
    args = parser.parse_args()
    
    # create 'logs' folder
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # create a folder for current experiment
    folder_path = os.path.join(os.getcwd(), 'logs', args.filepath)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("folder_path: ", folder_path)

    # number of classes in complete training
    n_classes = 0
    if args.training_type == 'class':
        n_classes = args.n_tasks * args.task_size
    elif args.training_type == 'domain':
        n_classes = args.task_size

    network = create_class_incremental_model(args.unit_size, args.units, 
                                             n_classes, args.radius, args.learning_rate, args.variance, 
                                             args.variance_alpha, 
                                             args.tau_radius, args.tau_lr)
    
    # load test samples
    test_samples = dataloader.loadNistTestData("../data/" + args.dataset, 
                                               args.training_type, 
                                               args.n_tasks, 
                                               args.task_size)
    
    final_accuracies = tf.constant([], dtype=tf.float32, 
                                   shape=(0, args.n_tasks))

    b = util.getRandomAccuracy(network, 
                               test_samples, 
                               args.n_tasks,
                               args.task_size, 
                               args.training_type)
    
    timeStep = 0
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
        
        # fit/train the model on train samples
        network.fit(train_samples, 
                    folder_path, 
                    index, 
                    args.task_size, 
                    args.training_type) #, 'vanilla', timeStep)
        timeStep += len(train_samples)

        test_config = network.getConfig()
        test_model = testClass(test_config['som'], 
                    test_config['shapeX'], 
                    test_config['shapeY'], 
                    test_config['unitsX'], 
                    test_config['unitsY'], 
                    test_config['class_count'], 
                    n_classes)
        test_model.setPMI()
        
        prediction_count = tf.zeros(args.n_tasks, dtype=tf.int32)
        label_count = tf.zeros(args.n_tasks, dtype=tf.int32)

        for task_index, samples in enumerate(test_samples):
            for sample in samples:
                feature_map = test_model.layer1(network.som, 
                                            sample.getImage())
                bmu = test_model.layer2(feature_map)
                output = tf.math.argmax(test_model.get_bmu_PMI(bmu))
                label = util.getTrainingLabel(sample.getLabel(),
                                            args.task_size,
                                            args.training_type)
                if tf.equal(output, label):
                    prediction_count = tf.tensor_scatter_nd_add(prediction_count, [[task_index]], [1])
                label_count = tf.tensor_scatter_nd_add(label_count, [[task_index]], [1])
        
        task_accuracy = tf.cast(prediction_count / label_count, 
                                dtype=tf.float32)
        print('task_accuracy: ', task_accuracy)
        final_accuracies = tf.concat([final_accuracies, 
                                      tf.reshape(task_accuracy, [1, -1])], 
                                      axis=0)

        print('final accuracies: ', final_accuracies)

    config = network.getConfig()
    pkl.dump(config, 
                open(os.path.join(folder_path, 'model_config.pkl'), 'wb'))
    del network
    
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

    