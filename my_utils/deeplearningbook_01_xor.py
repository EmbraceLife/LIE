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
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
# y = np.array([[0],[1],[1],[0]])
# it is better to be replaced with true target function ^ or xor function
y = X[:,0] ^ X[:,1] # ^ is xor function

# deeplearning book example on page
model = Sequential()
model.add(Dense(2, input_dim=2)) # 2 nodes reaches 50% accuracy, 8 nodes 100%
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid')) # output shaped by sigmoid
model.summary() # see each layer of a model

# use Model instead of Sequential
input_tensor = Input(shape=(2,))
hidden = Dense(2, activation='relu')(input_tensor)
output = Dense(1, activation='sigmoid')(hidden) # output shaped by sigmoid
model1 = Model(inputs=input_tensor, outputs=output)
model1.summary() # see each layer of a model

sgd = SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
model1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X, y, batch_size=1, epochs=10) # accuracy 0.75
model1.fit(X, y, batch_size=1, epochs=10) # accuracy 0.75

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
print("error": error)
"""
[[ 0.0033028 ]
 [ 0.99581173]
 [ 0.99530098]
 [ 0.00564186]]
"""
