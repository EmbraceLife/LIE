
#############################################
# convert batch_iterators into large arrays

from tensorflow.contrib.keras.python.keras.preprocessing.image import DirectoryIterator
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator



import numpy as np
# turn images in folders to DirectoryIterator then to arrays
def iterator2array(data_dir):

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
# try out
#
# get all images of a folder into a large array rather than batches
data_path_train = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/train"
train_img_array, train_lab_array = iterator2array(data_path_train)
