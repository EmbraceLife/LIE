###########################

"""
Inputs:
1. images folders for train, valid, test sets
2. settings: shuffle=True, batch_size=32

return:
1. train_batches, val_batches, test_batches

Steps:
1. import DirectoryIterator, ImageDataGenerator
2. get image folders ready
3. set target_size, color_mode, class_mode, batch_size, shuffle, data_format
"""
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


# img, lab = train_batches.next()
