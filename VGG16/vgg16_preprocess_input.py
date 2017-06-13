import tensorflow as tf
import numpy as np

## get preprocess_input function
prepInput = tf.contrib.keras.applications.vgg16.preprocess_input

# create 4D fake data
### Note
# it seems the proper shape for vgg16 is (batch, width, height, channel)
img_batch = np.random.random((1, 8, 8, 3))

# explore source code of tf.contrib.keras.applications.vgg16.preprocess_input
processed = prepInput(img_batch)
