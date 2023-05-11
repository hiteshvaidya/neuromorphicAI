"""
testClass.py

description: Use CosineDistanceLayer for inference
version: 1.0
author: Manali Dangarikar

"""

# import libraries
import tensorflow as tf
import os
from CosineDistanceLayer import CosineDistanceLayer
from MaxLayer import MaxLayer
import dataloader
from cnn2snn import check_model_compatibility
import pickle

with open('C:/Users/USER/source/repos/som/som/logs/trial-1/model_config.pkl', 'rb') as file:
    som_model = pickle.load(file)

class testClass(tf.keras.Model):
    """
    This class forms the network for the operation of SOM

    :param tf: tensorflow
    :type tf: tensorflow layer object
    """
    def __init__(self, som, shapeX, shapeY, unitsX, unitsY, class_count, n_classes):
        """
        constructor

        :param imgSize: number of pixels in a row and column of the input image or a unit in the SOM
        :type imgSize: int
        :param n_units: number of units in the SOM
        :type n_units: int
        """
        super(testClass, self).__init__()

        # Total number of pixels in a row and column of SOM
        self.shapeX = shapeX
        self.shapeY = shapeY

        # Total number of units in the SOM
        self.unitsX = unitsX
        self.unitsY = unitsY

        # randomly initialize pixels of the SOM
        #self.som = tf.random.normal([self.shapeX, self.shapeY],
        #                            0.1, 0.3)
        ## clip values between 0 and 1
        #self.som = tf.clip_by_value(self.som, 0.0, 1.0)
        self.som = som

        # Initialize the matrix for predicted class
        self.predicted_class = tf.ones([self.unitsX, self.unitsY]) * (-1.0)
        
        # class_count for every unit i.e. how many number of times a unit was selected as BMU for every class in the dataset
        self.class_count = tf.zeros([self.unitsX, self.unitsY, n_classes])
        self.class_count = class_count

        # Declare the layers of the network
        self.layer1 = CosineDistanceLayer(self.shapeX * self.shapeY, self.unitsX)
        self.layer2 = MaxLayer(self.unitsX)

    # @tf.function
    def InferencePass(self, x):
        """
        Inference pass through the network

        :param x: input image
        :type x: matrix of float values
        """
        x = self.layer1(self.som, x)
        x = self.layer2(x)
        print("max value:", x)
        return x
        
    def getPredictedClass(self, x):
        predictedClass = tf.gather(tf.gather(self.predicted_class, x[0]), x[1])
        return predictedClass

    def getAccuracy(self, y_pred, y_test):
        correct_predictions = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_test), tf.float32))
        accuracy = correct_predictions / tf.cast(tf.shape(y_test)[0], tf.float32)
        return accuracy

    def setPredictedClass(self):      
        # Update the predicted class for each unit
        a = tf.reduce_sum(self.class_count, axis=-1)
        indexes = tf.where(a == 0)
        labels = tf.cast(tf.argmax(self.class_count, axis=-1), dtype=tf.float32)
        values = -1.0 * tf.ones(indexes.shape[0])
        self.predicted_class = tf.tensor_scatter_nd_update(labels, indexes, values)

    
if __name__ == '__main__':
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    som = som_model['som']
    shapeX = som_model['shapeX']
    shapeY = som_model['shapeY']
    unitsX = som_model['unitsX']
    unitsY = som_model['unitsY']
    class_count = som_model['class_count']
    
    test = testClass(som, shapeX, shapeY, unitsX, unitsY, class_count, 10)
    test.setPredictedClass()
    print(test.predicted_class)
    # print(test.class_count)

    print("Model compatible for Akida conversion:",
      check_model_compatibility(test))
    ## Load the data
    #(x_train, y_train), (x_test, y_test) = dataloader.loadmnist()


    ## Declare object of test class using som matrix and predicted_class created during training
    #test = testClass(som_model, predicted_class)

    ## Infer for x_test[0]
    ## step 1: get the best matching unit for the input test sample
    #bmu = test.InferencePass(x_test[0])

    ## step 2: get the corresonding predicted class for the above bmu
    #y_pred = tf.constant(-1, shape=y_test.shape)
    #y_pred = tf.tensor_scatter_nd_update(y_pred, [[0]], [test.getPredictedClass(bmu)])

    ## step 3: calculate accuracy
    #runningAccuracy = test.getAccuracy(y_pred, y_test)
