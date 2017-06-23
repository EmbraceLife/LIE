"""
### Summary
- import wp model from previous source file
- train it with some pre-set hyperparameters
- best model during training is saved
"""

from keras.callbacks import TensorBoard, ModelCheckpoint
from prep_data_03_stock_04_load_saved_train_valid_test_features_targets_arrays import train_features, train_targets, valid_features, valid_targets, test_features, test_targets

from build_model_03_stock_02_build_compile_model_with_wp_init import wp
import numpy as np

# input_shape (window, num_indicators)
nb_epochs = 10
batch_size = 512
best_model_in_training = "/Users/Natsume/Downloads/DeepTrade_keras/models_save_load/best_model_in_training"



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
