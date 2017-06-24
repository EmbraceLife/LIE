"""
Inputs:
1. input_shape; 2. learning_rate; 3. n_layers; 4. n_hidden; 5. rate_dropout

Return:
1. a built and compiled model (not trained yet)

### Summary
- use WindPuller class to create a model and compile it
- a wp model can be imported from this source code

### Steps:
- import WindPull class
- set hyperparameters
- run __init__() to create and compile a model
"""


from build_model_03_stock_01_WindPuller_class_can_build_fit_evaluate_predict_model import WindPuller

# input_shape (window, num_indicators)
input_shape = (30, 61)
lr = 0.002
n_layers = 1
n_hidden=16
rate_dropout = 0.3

# build a model and compile it with WindPull.__init__()
wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)

wp.model.summary()
