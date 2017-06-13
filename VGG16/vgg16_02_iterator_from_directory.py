from tensorflow.contrib.keras.python.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

from vgg16_09_save_load_large_array import  bz_save_array, bz_load_array # np_save, np_load, pk_save, pk_load, idx_save, idx_load,

from vgg16_02_array_from_batches_from_folder import iterator2array

###########################
# from images in folders to image_batch_iterator
###########################
data_path_train = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/train"

train_batches = DirectoryIterator(directory = data_path_train,
							   image_data_generator=ImageDataGenerator(),
							   target_size=(224, 224),
							   color_mode = "rgb", # add up to (224,224,3)
							#    classes=["dogs", "cats"],
							   class_mode="categorical", # binary for only 2 classes
							   batch_size=32,
							   shuffle=True,
							   seed=123,
							   data_format="channels_last"
							#    save_to_dir
							#    save_prefix
							#    save_format
							   )
img, lab = train_batches.next()

data_path_val = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/valid"
val_batches = DirectoryIterator(directory = data_path_val,
							   image_data_generator=ImageDataGenerator(),
							   target_size=(224, 224),
							   color_mode = "rgb", # add up to (224,224,3)
							#    classes=["dogs", "cats"],
							   class_mode="categorical",
							   batch_size=32,
							   shuffle=True,
							   seed=123,
							   data_format="channels_last"
							#    save_to_dir
							#    save_prefix
							#    save_format
							   )

data_path_test = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/test"
test_batches = DirectoryIterator(directory = data_path_test,
							   image_data_generator=ImageDataGenerator(),
							   target_size=(224, 224),
							   color_mode = "rgb", # add up to (224,224,3)
							#    classes=["dogs", "cats"],
							   class_mode=None, # unknown about classes
							   batch_size=32,
							   shuffle=True,
							   seed=123,
							   data_format="channels_last"
							#    save_to_dir
							#    save_prefix
							#    save_format
							   )



trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"

###########################
# from images in folders to image_batch_iterator to large arrays
###########################
# convert batch_iterators into arrays
train_img_array, train_lab_array = iterator2array(data_path_train)
val_img_array, val_lab_array = iterator2array(data_path_val)
test_img_array, test_lab_array = iterator2array(data_path_test)

################################################
# save and load each array using bcolz
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
