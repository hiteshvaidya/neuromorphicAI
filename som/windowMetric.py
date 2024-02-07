import tensorflow as tf
from testSOM import testClass
import pickle as pkl
import dataloader
import os
import argparse
from network import Network
import numpy as np

class Metric:

    def __init__(self, folder_path):
        self.som_buffer = dataloader.loadModel(os.path.join(folder_path, "som_buffer.pkl"))
        self.running_variance_buffer = dataloader.loadModel(os.path.join(folder_path, "running_variance_buffer.pkl"))

    def create_class_incremental_model(self, unit_size, units, n_classes, radius, learning_rate, variance, variance_alpha, tau_radius, tau_lr):
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

    def getDifferenceMagnitude(self, matrix):
        difference = []
        for index in range(matrix.shape[0]):
            temp = matrix - matrix[index]
            difference.append(temp * temp)
        
        difference = tf.stack(difference)
        size = difference.shape[0]
        difference = tf.math.sqrt(difference) #/ size)
        return difference

    def getSpike(self):
        difference_magnitude = self.getDifferenceMagnitude(self.running_variance_buffer)
        mean_difference = tf.math.reduce_mean(difference_magnitude)

        return mean_difference
    
    def getWindows(self, threshold):
        difference_magnitude = self.getDifferenceMagnitude(self.running_variance_buffer)
        indices = tf.where(difference_magnitude > threshold)
        return indices
    
    def getBMU(self, test_model, index, test_sample):
        
        feature_map = test_model.layer1(self.som_buffer[index], 
                                    test_sample.getImage())
        bmu = test_model.layer2(feature_map)
        
    def getDeltaDifference(self, test_model, test_sample, index_m, index_n):
        bmu_m = self.getBMU(test_model, index_m, test_sample)
        bmu_n = self.getBMU(test_model, index_n, test_sample)

        return (bmu_m - bmu_n)
    
    def getDelta(self, indexLB, indexUB, test_model, test_samples):
        deltas = []
        for cursor1 in range(indexLB, indexUB):
            for cursor2 in range(cursor1, indexUB):
                for sample in test_samples:
                    value = self.getDeltaDifference(test_model, sample,
                                                    cursor1, cursor2)
                    deltas.append(value)
        
        max_difference = max(deltas)

        return max_difference
    
    def windowedPlasticity(self, data_path, test_model, n_tasks, task_size):
        mean_threshold = self.getSpike()
        indices = self.getWindows(mean_threshold)
        print("number of indices for window = ", len(indices))

        windowPlasticity = 0

        for task in range(n_tasks):
            test_samples = []
            for n_class in range(task*task_size, task*(1 + task_size)):
                class_samples = dataloader.loadSplitData(data_path, n_class)
                test_samples.extend(class_samples)
            
            delta_values = []
            for index in range(len(indices)-1):
                delta_values.append(self.getDelta(indices[index], 
                                                  indices[index+1], 
                                                  test_model, 
                                                  class_samples))
            windowPlasticity += max(delta_values)
        
        windowPlasticity /= (n_tasks * task_size)

        print("window plasticity = ", windowPlasticity)

        return windowPlasticity

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=None, required=True, help="dataset choice")
    parser.add_argument("-lp", "--log_path", type=str, default=None, required=True, help="logs folder path")
    parser.add_argument("-nt", "--n_tasks", type=int, default=None, required=False, help="number of tasks")
    parser.add_argument("-ts", "--task_size", type=int, default=1, required=False, help="number of classes")
    parser.add_argument("-t", "--training_type", type=str, default=None, required=True, help="class or domain")
    args = parser.parse_args()


    

    # number of classes in complete training
    n_classes = 0
    if args.training_type == 'class':
        n_classes = args.n_tasks * args.task_size
    elif args.training_type == 'domain':
        n_classes = args.task_size

    distance_type = "cosine"

    test_config = dataloader.loadModel(os.path.join(args.log_path, "model_config.pkl"))
    test_model = testClass(test_config['som'], 
                    test_config['shapeX'], 
                    test_config['shapeY'], 
                    test_config['unitsX'], 
                    test_config['unitsY'], 
                    test_config['class_count'], 
                    n_classes, distance_type)
    test_model.setPMI()
    