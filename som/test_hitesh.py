

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
from tensorflow.keras import layers

with open('C:/Users/USER/source/repos/som/som/logs/trial-1/model_config.pkl', 'rb') as file:
    som_model = pickle.load(file)

class MyCustomLayer(layers.Layer):
    def __init__(self, som_units, predicted_labels, **kwargs):
        super(MyCustomLayer, self).__init__(**kwargs)
        self.weight_matrix = som_units
        self.predicted_labels = predicted_labels

    def build(self, input_shape):
        # Define the weights or any additional variables needed for your layer
        self.my_variable = self.add_weight(shape=(input_shape[1],), initializer='random_normal', trainable=True)

    def call(self, inputs):
        # Reshape the input image to (1, 28*28) for batch multiplication
        input_image = tf.reshape(inputs, (1, -1))

        # Reshape the weight matrix to have submatrices of shape (28*28, 28*28)
        weight_submatrices = tf.image.extract_patches(
            images=tf.expand_dims(self.weight_matrix, axis=0),
            sizes=[1, 28, 28, 1],
            strides=[1, 28, 28, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        print("weight_submatrices: ", weight_submatrices.shape)
        weight_submatrices = tf.reshape(weight_submatrices, (-1, 28*28))

        # Compute the cosine similarity between the input image and each weight submatrix
        similarity = tf.keras.losses.CosineSimilarity(axis=1)(input_image, weight_submatrices)

        # Find the index of the weight submatrix with the maximum cosine similarity
        argmax_index = tf.argmax(similarity)

        # Get the corresponding value from the predicted_labels matrix
        predicted_label = self.predicted_labels[argmax_index / self.predicted_labels.shape[0], argmax_index % self.predicted_labels[1]]

        return predicted_label


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

    # Define the shape of the input image
    image_shape = (28, 28)

    # Define the shape of the feature map
    feature_map_shape = (560, 560)

    # Define the input layer for the input image
    input_image = layers.Input(shape=image_shape)

    # Define the input layer for the feature map
    feature_map = layers.Input(shape=feature_map_shape)
    
    # Reshape the feature map into submatrices of size (28, 28)
    submatrices = layers.Reshape((feature_map_shape[0]//image_shape[0],
                                  feature_map_shape[1]//image_shape[1],
                                  image_shape[0],
                                  image_shape[1]))(feature_map)

    # Flatten the input image
    input_image_flat = layers.Flatten()(input_image)
    input_image_flat_reshaped = tf.expand_dims(input_image_flat, axis=1)

    # Flatten the submatrices of the feature map
    submatrices_flat = layers.Reshape((-1, image_shape[0]*image_shape[1]))(submatrices)

    # Calculate cosine similarity using dot product and L2 normalization
    dot_product = layers.Dot(axes=2, normalize=True)([input_image_flat_reshaped, submatrices_flat])

    # Reshape the cosine similarity results back to the original submatrices shape
    cosine_similarity = layers.Reshape((feature_map_shape[0]//image_shape[0],
                                        feature_map_shape[1]//image_shape[1]))(dot_product)

    # Get the argmax on the indices
    argmax_indices = layers.Lambda(lambda x: tf.argmax(x, axis=-1))(cosine_similarity)
    argmax_indices = tf.reshape(argmax_indices, [-1, 20, 1])

    # Gather the values from the predicted_labels matrix based on the argmax indices
    predicted_labels = layers.Input(shape=(feature_map_shape[0]//image_shape[0],
                                           feature_map_shape[1]//image_shape[1]))  # Placeholder for the predicted_labels matrix
    selected_values = layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([predicted_labels, argmax_indices])

    inference_layer = MyCustomLayer(som, predicted_labels=test.predicted_class)
    output = inference_layer(input_image)
    # Create the Keras model
    # model = tf.keras.Model(inputs=[input_image, feature_map, predicted_labels], outputs=selected_values)
    model = tf.keras.Model(inputs=input_image, outputs=output)

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
