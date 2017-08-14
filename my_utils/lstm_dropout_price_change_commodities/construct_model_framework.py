"""
deep_trader_model_Sequential

Uses:
1. WindPuller class
	- __init__: build a model and compile
	- fit: to train
	- evaluate:
	- predict
	- save
	- load_model

"""
import tensorflow as tf
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout, initializers, Input
# from renormalization import BatchRenormalization
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.models import load_model
from keras.initializers import Constant
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.backend.tensorflow_backend import _to_tensor

#######################
# 如何构建自己的激励函数和损失函数：将下面两个函数直接插入相关源代码文档中最方便
#######################
def relu_limited(x, alpha=0., max_value=1.): # won't be used here
    return K.relu(x, alpha=alpha, max_value=max_value)

get_custom_objects().update({'custom_activation': Activation(relu_limited)})

###############

def risk_estimation(y_true, y_pred): # won't be used here
    return -100. * K.mean((y_true - 0.0002) * y_pred)


class WindPuller(object):

    def __init__(self, input_shape, lr=0.01, n_layers=2, n_hidden=8, rate_dropout=0.2, loss='mse'): # risk_estimation, risk_estimation_bhs

        print("initializing..., learing rate %s, n_layers %s, n_hidden %s, dropout rate %s." %(lr, n_layers, n_hidden, rate_dropout))

		# build a model with Sequential()
        self.model = Sequential()

		# todo: ask why dropout on input layer?
        self.model.add(Dropout(rate=rate_dropout, input_shape=(input_shape[0], input_shape[1])))

		# build a number of LSTM layers
        for i in range(0, n_layers - 1):
            self.model.add(LSTM(n_hidden * 4, return_sequences=True, activation='tanh',
                                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal', bias_initializer='zeros',
                                dropout=rate_dropout, recurrent_dropout=rate_dropout))
		# add another LSTM layer
        self.model.add(LSTM(n_hidden, return_sequences=False, activation='tanh',
                                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal', bias_initializer='zeros',
                                dropout=rate_dropout, recurrent_dropout=rate_dropout))


        self.model.add(Dense(1, kernel_initializer=initializers.glorot_uniform()))
        # self.model.add(Activation('sigmoid')) # no more sigmoid as last layer needed


		# compile model
        opt = RMSprop(lr=lr)
        self.model.compile(loss=loss,
                      optimizer=opt,
                      metrics=['accuracy'])

    def fit(self, x, y, batch_size=32, nb_epoch=100, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):
        self.model.fit(x, y, batch_size, nb_epoch, verbose, callbacks,
                       validation_split, validation_data, shuffle, class_weight, sample_weight,
                       initial_epoch, **kwargs)

    def save(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)
        return self

    def evaluate(self, x, y, batch_size=32, verbose=1,
                 sample_weight=None, **kwargs):
        return self.model.evaluate(x, y, batch_size, verbose,
                            sample_weight, **kwargs)

    def predict(self, x, batch_size=32, verbose=0):
        return self.model.predict(x, batch_size, verbose)

# """
# test model building
# """
# wp = WindPuller(input_shape = (30, 61))
# wp.model.summary()
