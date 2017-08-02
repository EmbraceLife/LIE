"""
instantiate_deep_trader_model
- set parameters
- build a model object
- check its summary

Uses:
1. run this file to build and compile a WindPuller model

Inputs:
1. input_shape; 2. learning_rate; 3. n_layers; 4. n_hidden; 5. rate_dropout

Return:
1. a built and compiled model (not trained yet)

"""

from construct_model_framework import WindPuller

# input_shape (window, num_indicators)
input_shape = (30, 61) # (30, 61) original or (30, 24) for QCG
lr = 0.001 # first floyd_model2_1000 used 0.002
n_layers = 1
n_hidden=16
rate_dropout = 0.3

# build a model and compile it with WindPull.__init__()
wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)

wp.model.summary()
