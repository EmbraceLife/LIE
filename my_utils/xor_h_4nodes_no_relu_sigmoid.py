"""
grokking_deep_learning_correlation
xor_example
print_number_format_control
plot_ax_title_size
plot_figure_suptitle
subplot_structure
plot_weights_curves_of_a_model

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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

X = np.array([[0,0],[0,1],[1,0],[1,1]])
# y = np.array([[0],[1],[1],[0]])
# it is better to be replaced with true target function ^ or xor function
y = X[:,0] ^ X[:,1] # ^ is xor function


#######################################################
# Sequential can specify more activations layer from normal layers than Model
#######################################################
# check whether 3 models are the same or not
model = Sequential()
model.add(Dense(4, input_dim=2, name='dense1')) # 2 nodes reaches 50% accuracy, 8 nodes 100%
# model.add(Activation('relu', name='dense1_act'))
model.add(Dense(1, name='dense2'))
# model.add(Activation('sigmoid', name='dense2_act')) # output shaped by sigmoid
model.summary() # see each layer of a model


# use Model instead of Sequential
input_tensor = Input(shape=(2,), name='input')
hidden = Dense(4, activation='linear', name='dense1_relu')(input_tensor)
output = Dense(1, activation='linear', name='dense2_sigm')(hidden) # output shaped by sigmoid
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
		np.array([[ 0.21884251,  0.37195587,  0.95793033,  0.37305808],
				[ 0.62441111,  0.79077578, -0.79303694,  0.8752284 ]], dtype='float32'),
		np.array([ 0.,  0.,  0.,  0.], dtype='float32'),
		np.array([[ 1.0339992 ],
					[-0.56100774],
					[ 0.35092974],
					[-0.34449542]], dtype='float32'),
	   np.array([ 0.], dtype='float32')]
model.set_weights(init_weights)

## prepare a list for each node's weights
w_1_1_1 = []
w_1_1_2 = []
w_1_2_1 = []
w_1_2_2 = []
w_1_3_1 = []
w_1_3_2 = []
w_1_4_1 = []
w_1_4_2 = []
w_2_1 = []
w_2_2 = []
w_2_3 = []
w_2_4 = []
w_1_1_1.append(init_weights[0][0,0])
w_1_1_2.append(init_weights[0][1,0])
w_1_2_1.append(init_weights[0][0,1])
w_1_2_2.append(init_weights[0][1,1])
w_1_3_1.append(init_weights[0][0,2])
w_1_3_2.append(init_weights[0][1,2])
w_1_4_1.append(init_weights[0][0,3])
w_1_4_2.append(init_weights[0][1,3])
w_2_1.append(init_weights[2][0,0])
w_2_2.append(init_weights[2][1,0])
w_2_3.append(init_weights[2][2,0])
w_2_4.append(init_weights[2][3,0])
losses = []
accuracies = []



# use SGD optimizer with learning rate 0.1
sgd = SGD(lr=0.1)
# set loss function to be mse, print out accuracy
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
# set loss function to be binary_crossentropy, print accuracy
# model1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


#######################################################
# batch_size = 1, update weights every sample;
# batch_size = 100, update weights every 100 sample;
# without validation_split, all dataset are trained
# with validation_split=0.25, 25% dataset is saved for validation,rest for training; during training, there will be loss and val_loss, accu, and val_accu
# shuffle = True, all samples of training set will be shuffled for each epoch training
# epochs = 10, train with the entire dataset for 10 times
# hist = fit() will record a loss for each epoch
#######################################################

# hist1 = model.fit(X, y, batch_size=1, validation_split=0.25, epochs=10) # accuracy 0.75
# hist2 = model.fit(X, y, batch_size=1, epochs=1000) # accuracy 100%, loss keep dropping down to 0.0016

#######################################################
# save all weights, loss, accuracy changes
#######################################################
epochs = 1000
for epoch in range(epochs):
	print("epoch:", epoch)
	hist = model.fit(X, y, batch_size=1, epochs=1)
	print("save each and every weight element into list")
	w_1_1_1.append(model.get_weights()[0][0,0])
	w_1_1_2.append(model.get_weights()[0][1,0])
	w_1_2_1.append(model.get_weights()[0][0,1])
	w_1_2_2.append(model.get_weights()[0][1,1])
	w_1_3_1.append(model.get_weights()[0][0,2])
	w_1_3_2.append(model.get_weights()[0][1,2])
	w_1_4_1.append(model.get_weights()[0][0,3])
	w_1_4_2.append(model.get_weights()[0][1,3])
	w_2_1.append(model.get_weights()[2][0,0])
	w_2_2.append(model.get_weights()[2][1,0])
	w_2_3.append(model.get_weights()[2][2,0])
	w_2_4.append(model.get_weights()[2][3,0])
	print("save each and every loss and accuracy")
	losses.append(hist.history['loss'][0])
	accuracies.append(hist.history['acc'][0])


#######################################################
# plot all weights and losses, accuracies
#######################################################

plt.figure()

plt.suptitle("xor_h_4nodes_no_relu_no_sigmoid")

ax1 = plt.subplot2grid((6, 4), (0, 0), colspan=2, rowspan=2)  # stands for axes
ax1.plot(losses, c='blue', label='losses') # change index name
ax1.legend(loc='best')
ax1.set_title("end value: %.04f" % losses[-1], fontsize = 5)


ax2 = plt.subplot2grid((6, 4), (0, 2), colspan=2, rowspan=2)
ax2.plot(accuracies, c='red', label='acc')
ax2.legend(loc='best')
ax2.set_title("end value: %.04f" % accuracies[-1], fontsize=5)

ax3 = plt.subplot2grid((6, 4), (2, 0), colspan=1, rowspan=2)
ax3.plot(w_2_1, c='green', label='w_2_1')
ax3.legend(loc='best')
ax3.set_title("end value: %.04f" % w_2_1[-1], fontsize=5)

ax4 = plt.subplot2grid((6, 4), (2, 1), colspan=1, rowspan=2)
ax4.plot(w_2_2, c='green', label='w_2_2')
ax4.legend(loc='best')
ax4.set_title("end value: %.04f" % w_2_2[-1], fontsize=5)

ax5 = plt.subplot2grid((6, 4), (2, 2), colspan=1, rowspan=2)
ax5.plot(w_2_3, c='green', label='w_2_3')
ax5.legend(loc='best')
ax5.set_title("end value: %.04f" % w_2_3[-1], fontsize=5)

ax6 = plt.subplot2grid((6, 4), (2, 3), colspan=1, rowspan=2)
ax6.plot(w_2_4, c='green', label='w_2_4')
ax6.legend(loc='best')
ax6.set_title("end value: %.04f" % w_2_4[-1], fontsize=5)

ax7 = plt.subplot2grid((6, 4), (4, 0), colspan=1, rowspan=1)
ax7.plot(w_1_1_1, c='green', label='w_1_1_1')
ax7.legend(loc='best')
ax7.set_title("end value: %.04f" % w_1_1_1[-1], fontsize=5)

ax8 = plt.subplot2grid((6, 4), (4, 1), colspan=1, rowspan=1)
ax8.plot(w_1_2_1, c='green', label='w_2_2_1')
ax8.legend(loc='best')
ax8.set_title("end value: %.04f" % w_1_2_1[-1], fontsize=5)

ax9 = plt.subplot2grid((6, 4), (4, 2), colspan=1, rowspan=1)
ax9.plot(w_1_3_1, c='green', label='w_2_3_1')
ax9.legend(loc='best')
ax9.set_title("end value: %.04f" % w_1_3_1[-1], fontsize=5)

ax10 = plt.subplot2grid((6, 4), (4, 3), colspan=1, rowspan=1)
ax10.plot(w_1_4_1, c='green', label='w_2_4_1')
ax10.legend(loc='best')
ax10.set_title("end value: %.04f" % w_1_4_1[-1], fontsize=5)

ax11 = plt.subplot2grid((6, 4), (5, 0), colspan=1, rowspan=1)
ax11.plot(w_1_1_2, c='green', label='w_1_1_2')
ax11.legend(loc='best')
ax11.set_title("end value: %.04f" % w_1_1_2[-1], fontsize=5)

ax12 = plt.subplot2grid((6, 4), (5, 1), colspan=1, rowspan=1)
ax12.plot(w_1_2_2, c='green', label='w_1_2_2')
ax12.legend(loc='best')
ax12.set_title("end value: %.04f" % w_1_2_2[-1], fontsize=5)

ax13 = plt.subplot2grid((6, 4), (5, 2), colspan=1, rowspan=1)
ax13.plot(w_1_3_2, c='green', label='w_1_3_2')
ax13.legend(loc='best')
ax13.set_title("end value: %.04f" % w_1_3_2[-1], fontsize=5)

ax14 = plt.subplot2grid((6, 4), (5, 3), colspan=1, rowspan=1)
ax14.plot(w_1_4_2, c='green', label='w_1_4_2')
ax14.legend(loc='best')
ax14.set_title("end value: %.04f" % w_1_4_2[-1], fontsize=5)

plt.tight_layout()
plt.show() # use this to save manually can control the size of image
# plt.savefig("/Users/Natsume/Downloads/data_for_all/deeplearningbook/xor_h_4nodes_relu_sigmoid.png")



#######################################################
# make predictions
#######################################################
print("features:", X)
print("targets:", y)
print("predictions:", model.predict(X))

# features:
# [[0 0]
#  [0 1]
#  [1 0]
#  [1 1]]
# targets: [0 1 1 0]
# predictions:
# [[ 0.50276363]
#  [ 0.50332505]
#  [ 0.50126481]
#  [ 0.50182629]]
