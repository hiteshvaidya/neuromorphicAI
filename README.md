# neuromorphicAI
Class project for CMPE 789 - Neuromorphic Computing

To run the program use following command,
`python network.py -g <gpu_id> -u <n_units> -v <initial_running_variance> -va <running_variance_alpha> -r <initial_radius> -lr <initial_learning_rate> -fp <output_folder_name> -d <dataset_choice>`
<br />
example, <br />
`python network.py -g 0 -u 20 -v 0.5 -va 0.9 -r 1.5 -lr 0.05 -fp mnist_trial-1 -d mnist`

testClass.py: Nested functions, created a custom test model using tf.keras.model. This model has custom layers implemented using user defined utility functions to calculate
cosine distance layer and argmax. This model could not be converted into akida because akida expects a "functional" model whereas this was a "subclassed" model.

testing.py: Making custom layer model using functional API. Created a model using inheritance from tf.keras.layers. The input has three image: test image, trained SOM, predicted class matrix. This model could not be converted into akida model because akida expects only one Input Layer.

testing1.py: Using one Input layer and tensors for SOM and predicted class

customLayer_testing.py: Custom layer using functional API with a "custom" class. This model could not be converted into akida model because akida does not take a custom layer.

ConvLayerCosine_testing.py: created a conv2D layer for calculating cosine similarity to get the BMU. This model could not be converted into akida model due to runtime error while doing matrix multiplication.
