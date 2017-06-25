"""
model.evaluate():
	- Returns the loss value & metrics values for the model in test mode

Inputs:
- x: (images or features)
	- array
	- list of arrays (if multiple inputs required by the model)
	- dict (map names with arrays)
- y: (labels or targets)
	- array
	- list of arrays (if multiple inputs required by the model)
	- dict (map names with arrays)
- batch_size:
	- integer
	- number of samples per gradient update
	- set it could help easy computation?
- verbose: 0 or 1
- sample_weight:
	- array of weights
	- purpose: to measure the different contribution of different samples to loss and metrics


Note: for catsdogs, test dataset won't be good here, as its labels are unknown

Return:
- scalar loss, and/or metrics values
- model.metrics_names show me the names of the loss and metrics
"""


from prep_data_utils_01_save_load_large_arrays_bcolz_np_pickle_torch import bz_load_array

test_lab_array = bz_load_array("/Users/Natsume/Downloads/data_for_all/dogscats/results/test_lab_array")
val_img_array = bz_load_array("/Users/Natsume/Downloads/data_for_all/dogscats/results/val_img_array")
val_lab_array = bz_load_array("/Users/Natsume/Downloads/data_for_all/dogscats/results/val_lab_array")


#############################################
# load the latest pretrained model
from tensorflow.contrib.keras.python.keras.models import load_model

trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
vgg16_ft_epoch3 = load_model(trained_model_path+'/train_vgg16_again_model_7.h5')


#############################################
# evaluate only produce loss and accuracy:
loss_metrics = vgg16_ft_epoch3.evaluate(val_img_array, val_lab_array, verbose=1)
# x, y are both a single array or a single list of arrays;
# batch_size, default to be 32, used for easy flow of data
