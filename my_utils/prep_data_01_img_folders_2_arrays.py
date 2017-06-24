""""
Inputs:
1. image directory: include 2 or more subdirectories for images, eg. cats folder, dogs folder
2. setting: shuffle=False, batch_size=1

Return:
1. image_arrays: contain all images data
2. label_arrays: contain all images labels

### Steps
1. DirectoryIterator: to convert folders of images to image_iterator
2. ImageDataGenerator: to get methods for image transformations
3. DirectoryIterator.next: tranform images and return a batch or a single image or label as array
4. stack image array and label array on top of each other, to return a large image array and label array

### Experiment args of DirectoryIterator here
1. target_size
2. color_mode
3. class_mode
4. batch_size
5. shuffle
6. data_format
"""


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
			# classes=["dogs", "cats"], # folders can handle it
            # class_mode=None, # no label is included
            # class_mode='binary', # label 1D is included
            class_mode='categorical', # label 2D is included, one-hot encoding included, i think;
            batch_size=1,
            shuffle=False, # so that images and labels order can be matched by its original order in folders
            seed=123,
            data_format="channels_last") # put channel 3 at the end


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
