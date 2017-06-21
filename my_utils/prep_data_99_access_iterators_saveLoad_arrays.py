# inputs: directory and filename to save arrays


################################################
# access batch_iterator on train_set, validation set and test sets
################################################
from prep_data_02_img_folder_2_iterators import test_batches, train_batches, val_batches

################################################
# create and access train_set folder arrays, validation set folder arrays, and test set folder arrays
################################################


from prep_data_01_img_folders_2_arrays import train_img_array, train_lab_array, val_img_array, val_lab_array, test_img_array, test_lab_array




################################################
# save and load each array above using bcolz
################################################
from prep_data_98_funcs_save_load_large_arrays import  bz_save_array, bz_load_array

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
