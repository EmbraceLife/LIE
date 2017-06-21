"""
predict a number of samples with a pretrained model (use the latest trained model)
1. to predict a given number of batches, using batch_iterator
2. to predict a full dataset array, using a full dataset array
"""

# access test_batches iterator, val_batches iterator
from prep_data_02_img_folder_2_iterators import test_batches, val_batches
from prep_data_01_img_folders_2_arrays import test_img_array, train_img_array

#############################################
# load the latest pretrained model
from tensorflow.contrib.keras.python.keras.models import load_model

trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
vgg16_ft_epoch3 = load_model(trained_model_path+'/train_vgg16_again_model_3.h5')


#############################################
# predict_generator:
# 1. generate 3 batches of predictions
# 2. regardless whether 3 batches > an epoch or 3 batches > full_dataset or not
# 2. all batches of predictions will be row-bind, then return as output
preds_test_1_batch = vgg16_ft_epoch3.predict_generator(generator=test_batches,
								steps=1, # predict for 3 batches even though one epoch is less than 2 epochs
								max_q_size=10,
								workers=1,
								pickle_safe=False,
								verbose=2
								)

#############################################
# predict:
# 1. generate (full_num_samples/batch_size) num of batches of predictions
# 2. num_batches * batch_size + final_batch_samples == full number of samples
# 2. all batches of predictions will be row-bind, then return as output
preds_test_full = vgg16_ft_epoch3.predict(test_img_array, verbose=2)

preds_train_full = vgg16_ft_epoch3.predict(train_img_array, verbose=2)
# preds_train_full.shape
