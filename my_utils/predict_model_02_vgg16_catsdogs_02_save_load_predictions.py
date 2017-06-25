#############################################
# access save and load array functions
from prep_data_utils_01_save_load_large_arrays_bcolz_np_pickle_torch import bz_save_array, bz_load_array
# access predictions on testset as array
from predict_model_02_vgg16_catsdogs_01_predict_Batches_fullarray import preds_test_full, preds_train_full

# path for saving
trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"

# save and load
bz_save_array(trained_model_path+"/preds_test", preds_test_full)
bz_save_array(trained_model_path+"/preds_train", preds_train_full)

# preds_test = bz_load_array(trained_model_path+"/preds_test")
