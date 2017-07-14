"""
xor_example

- use (X, y) to train a network to imitate xor function, in reality most of time, we don't know what exact the true function we want to learn
- a model with 2 layers, 9 parameters in total, perform a xor function
- activation takes no parameters, make the previous layer output non-linear
- why activation function to make non-linear? world is non-linear
- how do we choose activation function: grad_explode and grade_diminish issue
- loss function, learning algo make 9 parameters update closer to true xor function
"""


from tensorflow.contrib.keras.python.keras.models import Sequential, Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Activation, Input
from tensorflow.contrib.keras.python.keras.optimizers import SGD
from tensorflow.contrib.keras.python.keras import backend as K
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
# y = np.array([[0],[1],[1],[0]])
# it is better to be replaced with true target function ^ or xor function
y = X[:,0] ^ X[:,1] # ^ is xor function


#######################################################
# Sequential can specify more activations layer from normal layers than Model
#######################################################
# check whether 3 models are the same or not
model = Sequential()
model.add(Dense(2, input_dim=2, name='dense1')) # 2 nodes reaches 50% accuracy, 8 nodes 100%
model.add(Activation('relu', name='dense1_act'))
model.add(Dense(1, name='dense2'))
model.add(Activation('sigmoid', name='dense2_act')) # output shaped by sigmoid
model.summary() # see each layer of a model


# use Model instead of Sequential
input_tensor = Input(shape=(2,), name='input')
hidden = Dense(2, activation='relu', name='dense1_relu')(input_tensor)
output = Dense(1, activation='sigmoid', name='dense2_sigm')(hidden) # output shaped by sigmoid
model1 = Model(inputs=input_tensor, outputs=output)
model1.summary() # see each layer of a model

"""
# use Model to split layer and activation
input_tensor = Input(shape=(2,))
hidden = Dense(2)(input_tensor)
relu_hid = K.relu(hidden)
dense_out = Dense(1)(relu_hid) # output shaped by sigmoid
sigmoid_out = K.sigmoid(dense_out)

# inputs and outputs must be layer tensor, not just tensor made from K
model2 = Model(inputs=input_tensor, outputs=sigmoid_out)

model2.summary() # see each layer of a model
"""


#######################################################
# Get initial weights and set initial weights
#######################################################
model.get_weights()
init_weights = [
		np.array([[ 0.61500782, -0.8923322 ],
				  [-0.83968955, -0.36485523]], dtype='float32'),
		np.array([-0.01146811,  0.        ], dtype='float32'),
		np.array([[ 1.19760191], [-0.74140114]], dtype='float32'),
		np.array([ 0.02545028], dtype='float32')]
model.set_weights(init_weights)





# use SGD optimizer with learning rate 0.1
sgd = SGD(lr=0.1)
# set loss function to be mse, print out accuracy
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
# set loss function to be binary_crossentropy, print accuracy
model1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


#######################################################
# batch_size = 1, update weights every sample;
# batch_size = 100, update weights every 100 sample;
# without validation_split, all dataset are trained
# with validation_split=0.25, 25% dataset is saved for validation,rest for training; during training, there will be loss and val_loss, accu, and val_accu
# shuffle = True, all samples of training set will be shuffled for each epoch training
# epochs = 10, train with the entire dataset for 10 times
# hist = fit() will record a loss for each epoch
#######################################################

hist1 = model.fit(X, y, batch_size=1, validation_split=0.25, epochs=10) # accuracy 0.75
hist2 = model.fit(X, y, batch_size=1, epochs=1000) # accuracy 0.75


# See how weights changes to make function more close to xor function
epochs = 5
for epoch in range(epochs):
	print("epoch:", epoch)
	model.fit(X, y, batch_size=1, epochs=1)
	print("Layer1 weights shape:")
	print(model.layers[0].weights)
	print("Layer1 kernel:")
	print(model.layers[0].get_weights()[0]) # each training, network step closer to xor function
	print("Layer1 bias:")
	print(model.layers[0].get_weights()[1])

print(model.predict(X))
print(model1.predict(X))
error = model.evaluate([X], [y])
print("error", error)
"""
[[ 0.0033028 ]
 [ 0.99581173]
 [ 0.99530098]
 [ 0.00564186]]
"""
