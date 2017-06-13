from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.engine.topology import Input
# from tensorflow.contrib.keras.python.keras.layers import Input
import numpy as np
import tensorflow as tf

# create np_dataset (3, 5)
np_feed_Input = np.arange(start=-5, stop=10, step=1, dtype='float32').reshape((3,5))

# create tensor_dataset (3,5)
tensor_feed_np = tf.constant(np_feed_Input)

##### dive into Input, InputLayer, Node, Layer

# only know shape for cols or features, not know number of sampels or batch_size, return placeholder tensor (?, 5)
input_shape = Input(shape=(5,), name='input_tensor1')

# know (num_samples, cols), return placeholder tensor with (3,5)
input_batch_shape = Input(batch_shape=(3, 5), name='input_tensor2')

# know only shape (5,) and input_tensor is given, return constant tensor
input_tensor_shape = Input(shape=(5,), tensor=tensor_feed_np, name='input_tensor3') # tensor can't be np.array

x = Input(shape=(32,), name='input_tensor4')

# first create Dense layer, then run __call__() func
y = Dense(16, activation='softmax')(x)

model = Model(x, y)
