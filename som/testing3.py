import tensorflow as tf
from cnn2snn import check_model_compatibility
import pickle
from tensorflow.keras import layers

with open('C:/Users/USER/source/repos/som/som/logs/trial-1/model_config.pkl', 'rb') as file:
    som_model = pickle.load(file)

som = som_model['som']
shapeX = som_model['shapeX']
shapeY = som_model['shapeY']
unitsX = som_model['unitsX']
unitsY = som_model['unitsY']
class_count = som_model['class_count']

# Define custom weights and biases for the convolutional layer
W_conv1 = tf.Variable(tf.random.normal([3, 3, 1, 400], stddev=0.1))
b_conv1 = tf.Variable(tf.random.normal([400], stddev=0.1))

# Create the CNN model
model = tf.keras.Sequential([
    # Input layer
    layers.InputLayer(input_shape=(28, 28, 1)),
    
    # Convolutional layer with custom weights and biases
    layers.Conv2D(
        filters=400,
        kernel_size=(28, 28),
        strides=(1, 1),
        padding='same',
        activation='relu',
        use_bias=True,
        kernel_initializer=tf.constant_initializer(W_conv1),
        bias_initializer=tf.constant_initializer(b_conv1),
        trainable=True,
    ),
    
    # Flatten layer
    layers.Flatten(),
    
    # Output layer
    layers.Dense(units=10, activation='softmax')
])

# Print the model summary
model.summary()


