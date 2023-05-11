
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
from cnn2snn import convert
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Input, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

with open('logs/trial-3/model_config.pkl', 'rb') as file:
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

def split_units(som, dtype=None):
    # Replace `custom_tensors` with your custom tensors of shape (num_kernels, kernel_height, kernel_width, input_channels)
    som = tf.expand_dims(tf.expand_dims(som, axis=0), axis=-1)
        # Reshape the weight matrix to have submatrices of shape (28*28, 28*28)
    kernels = tf.image.extract_patches(
        images=som,
        sizes=[1, 28, 28, 1],
        strides=[1, 28, 28, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    kernels = tf.squeeze(kernels)
    kernels = tf.reshape(kernels, [-1, 784])
    print("kernels: ", kernels.shape)
    # weight_submatrices = tf.reshape(kernels, (-1, 28, 28))

    # assert shape == kernels.shape, "Shape mismatch"
    # return K.variable(kernels, dtype=dtype)

    return kernels

def get_max_value(tensors):
    max_index = tensors[0]
    other_tensor = tensors[1]
    max_value = K.gather(other_tensor, max_index)
    return max_value

def cosine_similarity(x, y):
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.sum(x * y, axis=-1)

def get_max_value(tensors):
    max_index = tensors[0]
    other_tensor = tensors[1]
    max_value = K.gather(other_tensor, max_index)
    return max_value

    
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
    # print(test.class_count)

    # Define the shape of the input image
    image_shape = (28, 28, 1)

    # Define the shape of the feature map
    feature_map_shape = som.shape

    # Define the input layer for the input image
    input_layer = layers.Input(shape=image_shape)

    num_kernels = unitsX * unitsY  # Specify the number of kernels
    kernel_size = (28, 28)  # Specify the kernel size
    kernels = split_units(som)
    kernel_tensor = K.reshape(kernels, (28, 28, 1, num_kernels))
    # Define a custom kernel initializer using a lambda function
    kernel_initializer = lambda shape, dtype=None: K.constant(kernel_tensor)

    # Define a custom initializer class
    class CustomInitializer:
        def __call__(self, shape, dtype=None):
            return K.constant(kernel_tensor, dtype=dtype)

        def get_config(self):
            return {}
    
    # # Register the custom initializer with Keras custom objects
    # custom_objects = {'CustomInitializer': CustomInitializer}
    # Register the custom initializer with Keras custom objects
    tf.keras.utils.get_custom_objects()['CustomInitializer'] = CustomInitializer

    conv_layer = Conv2D(num_kernels, 
                        kernel_size, 
                        kernel_initializer=CustomInitializer(),
                        use_bias=False)(input_layer)

    flatten_layer = Flatten()(conv_layer)

    cosine_sim = cosine_similarity(input_layer, conv_layer)

    max_index = K.argmax(cosine_sim, axis=-1)

    predictions = tf.reshape(test.predicted_class, [-1])

    output = get_max_value([max_index, predictions])

    # Create the Keras model
    # model = tf.keras.Model(inputs=[input_image, feature_map, predicted_labels], outputs=selected_values)
    model = tf.keras.Model(inputs=input_layer, outputs=output)

    # Compile the model (add loss, optimizer, etc.)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    akida_model = None
    

    (x_train, y_train), (x_test, y_test) = dataloader.loadmnist()
    print("x_test: ", x_test.shape, x_test[0].shape)

    
    print("Model compatible for Akida convrsion:", 
          check_model_compatibility(model, input_is_image=True))
    akida_model = convert(model, 
                        file_path='logs/akida_model.amf',
                        input_is_image=True)
    akida_model.summary()


    # myModel = model(x_test[0], som, test.predicted_class)

    

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
