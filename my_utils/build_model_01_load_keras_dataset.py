"""
Load keras dataset 
"""
from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.utils import to_categorical

mnist_path = mnist.get_file(fname='mnist.npz', origin=None) # the dataset is already downloaded, no origin is needed
"""
(['def get_file(fname,\n',
  '             origin,\n',
  '             untar=False,\n',
  '             md5_hash=None,\n',
  '             file_hash=None,\n',
  "             cache_subdir='datasets',\n",
  "             hash_algorithm='auto',\n",
  '             extract=False,\n',
  "             archive_format='auto',\n",
  '             cache_dir=None):\n',
"""
(train_img, train_lab), (test_img, test_lab) = mnist.load_data(path=mnist_path)

# Convert labels to categorical one-hot encoding
train_lab_hot = to_categorical(train_lab, num_classes=10)
