
###############
"""
See the effect of BatchNormalization and Dropout
"""
from tensorflow.contrib.keras.python.keras.layers import Dropout, BatchNormalization, Input
from tensorflow.contrib.keras.python.keras.models import Model
import numpy as np
from tensorflow.contrib.keras.python.keras import backend as K

input_array_small = np.random.random((10,10))*2


input_tensor = Input(shape=(10,))
bn_tensor = BatchNormalization()(input_tensor)
dp_tensor = Dropout(0.7)(input_tensor)



#### for BatchNormalization, to see the effect of BatchNormalization
# test mode from Model method
model_bn = Model(input_tensor, bn_tensor)
bn_array = model_bn.predict(input_array_small)
# test and train mode from K.function method
k_bn = K.function([input_tensor, K.learning_phase()], [bn_tensor])
bn_array_test = k_bn([input_array_small, 0])[0]
bn_array_train = k_bn([input_array_small, 1])[0]

#### for Dropout, to see the effect
# test mode from Model method
model_dp = Model(input_tensor, dp_tensor)
dp_array = model_dp.predict(input_array_small)
# test and train mode from K.function method
k_dp = K.function([input_tensor, K.learning_phase()], [dp_tensor])
dp_array_test = k_dp([input_array_small, 0])[0]
dp_array_train = k_dp([input_array_small, 1])[0]

### both BatchNormalization and Droput have some basic operations prior to Normalization and Droping, Diving into the source when feeling so
