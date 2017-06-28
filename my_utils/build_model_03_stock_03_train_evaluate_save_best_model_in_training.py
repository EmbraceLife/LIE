"""
Uses:     run this file
1. to train the model built and compiled in previous source file
2. create a tensorboard for it
3. save best model during training
4. evaluate this model


Inputs:
1. nb_epochs;
2. batch_size;
3. best_model_in_training (path)

Return:
1. wp has been trained with new weights
2. a best model is saved onto the path
3. wp is evaluated with a loss and accuracy on test set

### Summary
- import wp model from previous source file
- train it with some pre-set hyperparameters
- best model during training is saved
- evaluate the model with test set
"""

from keras.callbacks import TensorBoard, ModelCheckpoint
from prep_data_03_stock_04_load_saved_train_valid_test_features_targets_arrays import train_features, train_targets, valid_features, valid_targets, test_features, test_targets

from build_model_03_stock_02_build_compile_model_with_WindPuller_init import wp
import numpy as np

# input_shape (window, num_indicators)
nb_epochs = 10
batch_size = 512
best_model_in_training = "/Users/Natsume/Downloads/data_for_all/stocks/best_model_during_training/best_model_in_training.h5"



# train model, and save the best model
wp.fit(train_features, train_targets, batch_size=batch_size,
	   nb_epoch=nb_epochs, shuffle=True, verbose=1,
	   validation_data=(valid_features, valid_targets),
	   callbacks=[TensorBoard(histogram_freq=1),
				  ModelCheckpoint(filepath=best_model_in_training, save_best_only=True, mode='min')])

# evaluate model
scores = wp.evaluate(test_features, test_targets, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
