###########################
# from images in folders to batch_iterators
# inputs args1: folders for train, valid, test sets
# inputs args2: for setting up DirectoryIterator()
# return: train_batches, val_batches, test_batches
###########################

from tensorflow.contrib.keras.python.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator


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
# img, lab = train_batches.next()

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
