"""
test_train_middle_layer_output_array_loaded_best_model
dropout_layer_behaviour


Example on middle layer output on train and test mode

- BatchNormalization layer output arrays on test and train mode

- Dropout layer output arrays on test and train mode
"""


from tensorflow.contrib.keras.python.keras.layers import Dropout, BatchNormalization, Input, Dense
from tensorflow.contrib.keras.python.keras.models import Model, Sequential, load_model
import numpy as np
from tensorflow.contrib.keras.python.keras import backend as K

input_array_small = np.random.random((500,10))*2
target_small = np.random.random((500,1))

input_tensor = Input(shape=(10,))
bn_tensor = BatchNormalization()(input_tensor)
dp_tensor = Dropout(0.7)(input_tensor)


#### Access BatchNormalization layer's output as arrays in both test, train mode

# test mode from Model method
model_bn = Model(input_tensor, bn_tensor)
bn_array = model_bn.predict(input_array_small)

# test and train mode from K.function method
k_bn = K.function([input_tensor, K.learning_phase()], [bn_tensor])
bn_array_test = k_bn([input_array_small, 0])[0]
bn_array_train = k_bn([input_array_small, 1])[0]

# are test mode the same? and test mode array differ from train mode array
(bn_array == bn_array_test).sum()
bn_array.shape # compare to see for equality
(bn_array == bn_array_train).sum() # total differ



#### Access Dropout layer's output as array in both test and train mode

# test mode from Model method
model_dp = Model(input_tensor, dp_tensor)
dp_array = model_dp.predict(input_array_small)

# test and train mode from K.function method
k_dp = K.function([input_tensor, K.learning_phase()], [dp_tensor])
dp_array_test = k_dp([input_array_small, 0])[0]
dp_array_train = k_dp([input_array_small, 1])[0]

# are test mode the same? and test mode array differ from train mode array
(dp_array == dp_array_test).sum()
dp_array.shape # compare to see for equality
(dp_array == dp_array_train).sum() # total differ


### both BatchNormalization and Droput have some basic operations prior to Normalization and Droping, Diving into the source when feeling so


#### Access middle layer output when First Dropout and then BatchNormalization

model_seq = Sequential()
model_seq.add(Dropout(0.3, input_shape=(10,)))
model_seq.add(BatchNormalization())
model_seq.add(Dense(1))
# check out weights before training
# model_seq.get_weights()

# compile and train
model_seq.compile(optimizer='SGD', loss='mse')

model_seq.fit(input_array_small, target_small, epochs=10)
model_seq.get_weights()
model_seq.save("to_delete.h5")
model_best = load_model("to_delete.h5")

###### compare two weights from two different training
# model_seq.get_weights()

###### check output
batchNorm_test = K.function([model_best.input, K.learning_phase()], [model_best.layers[-2].output])([input_array_small, 0])[0]
batchNorm_train = K.function([model_best.input, K.learning_phase()], [model_best.layers[-2].output])([input_array_small, 1])[0]


# dropout_layer_behaviour
## 下面代码帮助解释，dropout_rate = 0.3的含义，将30%的neuron权重参数设置为0
##### in test, there is 0 neuron output to be 0.0
drop_test = K.function([model_best.input, K.learning_phase()], [model_best.layers[-3].output])([input_array_small, 0])[0]
##### in train, there are 30% neuron output to be 1500 items of 0.0
drop_train = K.function([model_best.input, K.learning_phase()], [model_best.layers[-3].output])([input_array_small, 1])[0]
preds = model_best.predict(input_array_small)
