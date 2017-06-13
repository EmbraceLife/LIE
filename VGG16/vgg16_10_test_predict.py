# access test_batches iterator, val_batches iterator
from vgg16_iterator_from_directory import test_batches, val_batches


#############################################
# access load_model and load vgg16_2class model
from tensorflow.contrib.keras.python.keras.models import load_model

trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
vgg16_2class = load_model(trained_model_path+'/vgg16_2class.h5')
# this model modified and trained on dogs and cats


#############################################
# get preds_test: all samples of different bactches in test set are rbind together
preds_test = vgg16_2class.predict_generator(generator=test_batches,
								steps=3, # predict for 3 batches even though one epoch is less than 2 epochs
								max_q_size=10,
								workers=1,
								pickle_safe=False,
								verbose=2
								)

# print(preds_test)

#############################################
# get preds_val: all samples of different bactches in validation set are rbind together
preds_val = vgg16_2class.predict_generator(generator=val_batches,
								steps=3, # predict for 3 batches even though one epoch is less than 2 epochs
								max_q_size=10,
								workers=1,
								pickle_safe=False,
								verbose=2
								)

#############################################
# save large arrays using bcolz
from save_load_large_array import bz_save_array, bz_load_array

bz_save_array(trained_model_path+"/preds_test", preds_test)
bz_save_array(trained_model_path+"/preds_val", preds_val)
# preds_bc = bz_load_array(trained_model_path+"/preds_val")

#####################################
# steps = 1 and steps =3 make a differece to predict_generator?
