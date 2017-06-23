"""
### Summary
- use WindPuller class to create a model and compile it
- a wp model can be imported from this source code

### Steps:
- import WindPull class
- set hyperparameters
- run __init__() to create and compile a model
"""


from build_model_03_stock_01_WindPuller_init_fit_evaluate_predict import WindPuller

# input_shape (window, num_indicators)
nb_epochs = 10
batch_size = 512
input_shape = (30, 61)
lr = 0.002
n_layers = 1
n_hidden=16
rate_dropout = 0.3

# build a model and compile it with WindPull.__init__()
wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)

wp.model.summary()
