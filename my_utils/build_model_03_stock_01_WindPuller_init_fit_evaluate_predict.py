
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout, initializers
from renormalization import BatchRenormalization
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.models import load_model
from keras.initializers import Constant

"""
### Summary
- build a class with methods to create model, compile model, train model, evaluate model, predict model, save and load model
- WindPuller class can be imported from this source code 

### Steps
1. WindPuller.__init__(): build a model and compile it
2. WindPuller.fit(): train the model with x, y as two arrays, not iterators
3. WindPuller.save() and WindPuller.load_model(): save and load model with model pathname
4. WindPuller.evaluate(): x is array, not generator
5. WindPuller.test(): x is array, even with batch_size=32
"""

class WindPuller(object):

    def __init__(self, input_shape, lr=0.01, n_layers=2, n_hidden=8, rate_dropout=0.2, loss='risk_estimation'):

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

		# add a dense layer, with BatchRenormalization, relu_limited
        self.model.add(Dense(1, kernel_initializer=initializers.glorot_uniform()))
        # self.model.add(BatchNormalization(axis=-1, moving_mean_initializer=Constant(value=0.5),
        #               moving_variance_initializer=Constant(value=0.25)))
        self.model.add(BatchRenormalization(axis=-1, beta_init=Constant(value=0.5)))
        self.model.add(Activation('relu_limited'))

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
