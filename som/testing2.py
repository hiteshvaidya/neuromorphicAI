
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
# from cnn2snn import check_model_compatibility
import pickle
from tensorflow.keras import layers


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
        self.layer1 = CosineDistanceLayer(self.shapeX // self.unitsX,
                                            self.unitsX)
        self.layer2 = MaxLayer(self.unitsX)
        
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
    # print(test.predicted_class)
    # print(test.class_count)

    # Define the shape of the input image
    image_shape = (28, 28)

    # Define the shape of the feature map
    feature_map_shape = (560, 560)

    # Define the input layer for the input image
    input_image = tf.keras.Input(shape=image_shape, name='input_image')
    print(input_image.shape)
    # Define the variable for the feature map
    feature_map = tf.Variable(tf.zeros(feature_map_shape), name='feature_map')
    print(feature_map.shape)
    # Define the variable for the predicted labels
    predicted_labels = tf.Variable(tf.zeros((feature_map_shape[0]//image_shape[0], feature_map_shape[1]//image_shape[1])), name='predicted_labels')
    print(predicted_labels.shape)
    # Reshape the feature map into submatrices of size (28, 28)
    submatrices = tf.reshape(feature_map, (-1, feature_map_shape[0]//image_shape[0],
                                           feature_map_shape[1]//image_shape[1],
                                           image_shape[0],
                                           image_shape[1]))
    print(submatrices.shape)
    # Flatten the input image
    input_image_flat = tf.keras.layers.Flatten()(input_image)
    print(input_image_flat.shape)
    # input_image_flat_reshaped = tf.expand_dims(input_image_flat, axis=1)

    # Flatten the submatrices of the feature map
    submatrices_flat = tf.reshape(submatrices, (400, image_shape[0]*image_shape[1]))

    print(submatrices_flat.shape)
    # Calculate cosine similarity using dot product and L2 normalization
    # dot_product = tf.keras.layers.Dot(axes=2, normalize=True)([input_image_flat_reshaped, submatrices_flat])
    dot_product = tf.keras.layers.Dot(axes=-1, normalize=True)([input_image_flat, submatrices_flat])
    print(dot_product.shape)
    # Reshape the cosine similarity results back to the original submatrices shape
    cosine_similarity = tf.reshape(dot_product, (feature_map_shape[0]//image_shape[0],
                                                 feature_map_shape[1]//image_shape[1]))

    # Get the argmax on the indices
    argmax_indices = tf.keras.layers.Lambda(lambda x: tf.argmax(x, axis=-1))(cosine_similarity)
    argmax_indices = tf.reshape(argmax_indices, [-1, 20, 1])

    # Gather the values from the predicted_labels matrix based on the argmax indices
    selected_values = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([predicted_labels, argmax_indices])

    # Define the model
    model = tf.keras.Model(inputs=[input_image], outputs=selected_values)

    # Compile the model (add loss, optimizer, etc.)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    (x_train, y_train), (x_test, y_test) = dataloader.loadmnist()

    # myModel = model(x_test[0], som, test.predicted_class)

    print("Model compatible for Akida conversion:",
      check_model_compatibility(model))

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

