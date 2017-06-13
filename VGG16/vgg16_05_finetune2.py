# from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16
# vgg16=VGG16()


# to use Model() to create models
from tensorflow.contrib.keras.python.keras.models import Model
# to use Dense
from tensorflow.contrib.keras.python.keras.layers import Dense
# to use Adam optimizers
from tensorflow.contrib.keras.python.keras.optimizers import Adam

# # method2
# import tensorflow as tf
# import numpy as np
# Dense = tf.contrib.keras.layers.Dense
# Adam = tf.contrib.keras.optimizers.Adam

def finetune_2layers(vgg16, num, lr = 0.001):
	"""
		1. drop the last layer of current model;
		2. make all layers non-trainable;
		3. add a new Dense layer with 'num' output nodes, softmax activation, use the most previous immediate layer's output as this layer's input.
		4. build a new model with inputs as vgg16's input layer, outputs as the last dense layer;
		5. compile the new model

		Args:
			num (int) : Number of neurons in the Dense layer
		Returns:
			None
	"""
	# remove the latest layer
	vgg16.layers.pop()

	# check which layer is removed
	# vgg.summary()

	# make all layers non-trainable
	for layer in vgg16.layers: layer.trainable=False

	# units: dimension of output sapece;
	# Note: both vgg16.layers[-1].output and x0 are tensors, not layers
	x0 = Dense(units=100, activation='relu', name='last_dense')(vgg16.layers[-1].output)
	x = Dense(units=num, activation='softmax', name='predictions_2')(x0)

	# to build a new model, inputs==model's input layer|tensor, outputs == model's output tensor
	vgg16 = Model(inputs=vgg16.input, outputs=x)

	# compile this new model
	vgg16.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

	return vgg16

# vgg16_2layers = finetune_2layers(vgg16, num=2)
