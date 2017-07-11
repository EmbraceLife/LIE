"""
middle_layer_output_array_by_Model
middle_layer_output_array_by_K_function
middle_layer_output_array_test_mode
middle_layer_output_array_train_mode


### Access middle layers (in test mode or train mode)?

- middle_layer_tensor = Model(input_tenor, middle_layer_tensor).predict(input_array)

- middle_layer_tensor = Model(model.input, model.layers[3].output).predict(input_array)

- middle_layer_tensor = K.function([input_tensor], [output_tensor])([input_array])[0]

- middle_layer_tensor = K.function([input_tensor, K.learning_phase()], [output_tensor])([input_array, 0])[0] # for test mode

- middle_layer_tensor = K.function([input_tensor, K.learning_phase()], [output_tensor])([input_array, 1])[0] # for train mode

"""


"""
todo: K.learning_phase()???

One simple way is to create a new `Model` that will output the layers that you are interested in:
"""

from tensorflow.contrib.keras.python.keras.models import Model, Sequential
from tensorflow.contrib.keras.python.keras.layers import Input, Dense
from tensorflow.contrib.keras.python.keras import backend as K
import numpy as np



input_tensor = Input(shape=(100,), name="input_tensor")
inter_tensor = Dense(30, name="my_layer")(input_tensor)
final_tensor = Dense(30, name="final_layer")(inter_tensor)
model = Model(input_tensor, final_tensor)  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

# must compile before predict? No, but must compile before training
input_array1 = np.random.random((1000, 100))*9
input_tensor1 = K.constant(value=input_array1)
intermediate_output = intermediate_layer_model.predict(input_array1) # return array
intermediate_output1 = intermediate_layer_model(input_tensor1) # return tensor not array; tensor is no use and a long way to go to reach array

"""
Alternatively, you can build a Keras function that will return the output of a certain layer given a certain input, for example:
"""

from tensorflow.contrib.keras.python.keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
layer_output = get_3rd_layer_output([input_array1])[0] # [0] due to return a list



"""
Similarly, you could build a Theano and TensorFlow function directly.

Note that if your model has a different behavior in training and testing phase (e.g. if it uses `Dropout`, `BatchNormalization`, etc.), you will need
to pass the learning phase flag to your function:
"""
from tensorflow.contrib.keras.python.keras.layers import Dropout, BatchNormalization
input_tensor = Input(shape=(100,), name="input_tensor")
inter_tensor = Dense(300, name="my_layer")(input_tensor)
bn_tensor = BatchNormalization()(inter_tensor)
drop_tensor = Dropout(0.7)(bn_tensor)
final_tensor = Dense(30, name="final_layer")(drop_tensor)
model_dp_bn = Model(input_tensor, final_tensor)  # create the original model

# K.function can help distinct test_mode and training mode; as in training_mode, BatchNormalization is working, whereas in test_mode, BatchNormalization is not applied
get_3rd_layer_output = K.function([model_dp_bn.layers[0].input, K.learning_phase()], [model_dp_bn.layers[3].output])

# output in test mode = 0: no zeros in output
layer_output_0 = get_3rd_layer_output([input_array1, 0])[0]
layer_output_0.max()
layer_output_0.min()

# output in train mode = 1: lost of zeros in output
layer_output_1 = get_3rd_layer_output([input_array1, 1])[0]
layer_output_1.max()
layer_output_1.min()

# use a different method, we can only get test_mode, not training mode
model2 = Model(input_tensor, bn_tensor)
model2.predict(input_array1).max()
