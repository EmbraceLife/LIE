"""
Inputs:
1. pre-trained and weights loaded model
2. hyper-parameters for new layers added onto the above model
3. hyper-parameters (optimization and loss) for model compilation

Return:
1. a new compiled model

Steps:
1. import the pretrained and weight-loaded model
2. import methods and classes to add new layers and compile model
3. remove certain layers
4. add new layers
5. create a new model
6. compile this model
7. return this new model
"""

# to use official VGG16
# from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16

# use vgg16 model with access to all layer output tensor
from build_model_01_vgg16_Dive_01_source_vgg16 import VGG16
vgg16=VGG16()
# to use Model() to create models
from tensorflow.contrib.keras.python.keras.models import Model
# to use Dense
from tensorflow.contrib.keras.python.keras.layers import Dense
# to use Adam optimizers
from tensorflow.contrib.keras.python.keras.optimizers import Adam


def finetune(vgg16, num, lr = 0.001):
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

	# make all layers' kernerl, biases non-trainable
	for layer in vgg16.layers: layer.trainable=False

	# latest layer's output tensor
	last_output_tensor = vgg16.layers[-1].output

	x = Dense(units=100, activation='relu', name='last_dense')(last_output_tensor)
	last_second_tensor = x

	# add a new final layer
	x = Dense(units=num, activation='softmax', name='predictions')(x)
	final_layer_output_tensor = x

	# to build a new model with first and last output tensor
	vgg16 = Model(inputs=vgg16.input, outputs=x)

	# use object rather than string can customize the optimizer
	vgg16.compile(optimizer=Adam(lr=lr),
			loss='categorical_crossentropy', metrics=['accuracy'])

	return vgg16

# for classifying 2 classes, if DirectoryIterator use `class_mode='categorical'`
# we need 2 nodes in the last layer for output
vgg16_ft = finetune(vgg16, num=2) # if num = 2, class_mode="categorical"

# vgg16.summary()
# vgg16_ft.summary()
