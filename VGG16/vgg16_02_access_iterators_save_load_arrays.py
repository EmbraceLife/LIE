################################################
# access batch_iterator on train_set, validation set and test sets
################################################
from vgg16_02_from_img_directory_2_iterators import test_batches, train_batches, val_batches

################################################
# create and access train_set folder arrays, validation set folder arrays, and test set folder arrays
################################################


from vgg16_02_array_from_batches_from_folder import train_img_array, train_lab_array, val_img_array, val_lab_array, test_img_array, test_lab_array




################################################
# save and load each arrays using bcolz
################################################
from vgg16_09_save_load_large_array import  bz_save_array, bz_load_array # np_save, np_load, pk_save, pk_load, idx_save, idx_load,

trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"

bz_save_array(trained_model_path+"/train_img_array", train_img_array)
bz_save_array(trained_model_path+"/train_lab_array", train_lab_array)
bz_save_array(trained_model_path+"/val_img_array", val_img_array)
bz_save_array(trained_model_path+"/val_lab_array", val_lab_array)
bz_save_array(trained_model_path+"/test_img_array", test_img_array)
bz_save_array(trained_model_path+"/test_lab_array", test_lab_array)

# load the check the arrays
try_img_array = bz_load_array(trained_model_path+"/train_img_array")
try_lab_array = bz_load_array(trained_model_path+"/train_lab_array")


################################################
# # save and load array using np_save, np_load
# np_save(trained_model_path+"/train_img_array", train_img_array)
# try_img_array = np_load(trained_model_path+"/train_img_array.npy")


######################
# save and load with pickle
# pk_save(trained_model_path+"/train_img_array_pk", train_img_array)
# try_img_array = pk_load(trained_model_path+"/train_img_array_pk")


######################
# idx_save(trained_model_path+"/train_img_array_idx", train_img_array)
# train_img_idx = idx_load(trained_model_path+"/train_img_array_idx")
