
#############################################
# convert images of folders to batch_iterators into large arrays
# args1: directory contains two or more sub folders of images (cats, dogs)
# args2: target_size as (224, 224), convert original image size to target_size as we want
# args3: we decide color_mode: either rgb or grayscale,
# args4: class_mode: categorical mostly
# args5: batch_size: how many samples in each batch
# args6: data_format: (width, height, channel) or (channel, width, height)
# args7: shuffle: true or false, shuffle samples and then get batches

from tensorflow.contrib.keras.python.keras.preprocessing.image import DirectoryIterator
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator

import numpy as np



# turn images in folders to batch_iterator then to a single array
def img_folders_2_array(data_dir):

	# turn images of folders into batch iterators
    batch_iterator = DirectoryIterator(
            directory = data_dir,
            image_data_generator=ImageDataGenerator(),
            target_size=(224, 224),
            color_mode = "rgb", # add up to (224,224,3)
			#    classes=["dogs", "cats"],
            # class_mode=None, # no label is included
            # class_mode='binary', # label 1D is included
            class_mode='categorical', # label 2D is included, one-hot encoding included, i think;
            batch_size=1,
            shuffle=False, # so that images and labels order can be matched
            seed=123,
            data_format="channels_last")


    img_array = np.concatenate([batch_iterator.next()[0] for i in range(batch_iterator.samples)])
    lab_array = np.concatenate([batch_iterator.next()[1] for i in range(batch_iterator.samples)])

    return img_array, lab_array



##########
# get all images of a folder into a large array rather than batches
data_path_train = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/train"
train_img_array, train_lab_array = img_folders_2_array(data_path_train)

data_path_val = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/valid"
val_img_array, val_lab_array = img_folders_2_array(data_path_val)

data_path_test = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/test"
test_img_array, test_lab_array = img_folders_2_array(data_path_test)
