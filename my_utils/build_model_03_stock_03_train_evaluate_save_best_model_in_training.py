"""
train_deep_trader_model
predict_middle_layer_output_after_training


train_deep_trader_model
- get dataset ready
- train a number of epochs
- save trained and tensorboard log

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
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from prep_data_03_stock_04_load_train_valid_test_features_targets_arrays_from_files_for_train import train_features, train_targets, valid_features, valid_targets, test_features, test_targets
from keras.models import load_model
from build_model_03_stock_02_build_compile_model_with_WindPuller_init import wp
import numpy as np

# input_shape (window, num_indicators)
nb_epochs = 1 # set it large when train with floyd
batch_size = 512

########################################################################
#### store model locally or on floydhub
########################################################################
best_model_in_training = "/Users/Natsume/Desktop/best_model_in_training.h5" # local computer training

# best_model_in_training = "/output/during_best.h5" # store on floydhub

# train model, and save the best model
wp.fit(train_features, train_targets, batch_size=batch_size,
	   nb_epoch=nb_epochs, shuffle=True, verbose=1,
	   validation_data=(valid_features, valid_targets),
	   callbacks=[TensorBoard(histogram_freq=1,
	#    log_dir='/output/log'), # store on floydhub
	    log_dir="/Users/Natsume/Desktop/log"), # log/ used for saving locally, # /output/log used for saving log for floyd too
				  ModelCheckpoint(filepath=best_model_in_training, save_best_only=True, mode='min')])

# evaluate model
scores = wp.evaluate(test_features, test_targets, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

preds = wp.predict(test_features)
print(preds)

# get middle layer's output on test mode
batchNorm_test = K.function([wp.model.input, K.learning_phase()], [wp.model.layers[-2].output])([test_features, 0])[0]
# on train mode
batchNorm_test = K.function([wp.model.input, K.learning_phase()], [wp.model.layers[-2].output])([test_features, 1])[0]

wp.save("/Users/Natsume/Desktop/model_3000.h5") # saving locally
# wp.save("/output/exact_model2_1000.h5") # saving on floyd

model1 = load_model("/Users/Natsume/Desktop/model_3000.h5")
batchNorm_test = K.function([model1.input, K.learning_phase()], [model1.layers[-2].output])([test_features, 1])[0]
batchNorm_test = K.function([model1.input, K.learning_phase()], [model1.layers[-2].output])([test_features, 0])[0]
#### without output, the following floyd command is working
### make sure: /input/data_file_name, even though dataset saved in floyd/data/daniel/features_targets_train_val_test
# floyd run --data DJeKLuEpYqJPBYhxViyRfm --gpu "python build_model_03_stock_03_train_evaluate_save_best_model_in_training.py " --env keras

#### with output, the following could working
# floyd run --data DJeKLuEpYqJPBYhxViyRfm --gpu "python build_model_03_stock_03_train_evaluate_save_best_model_in_training.py > /output/output/model_3000.h5" --env keras
