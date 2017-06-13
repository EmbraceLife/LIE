
import numpy as np

# not sure what this function is trying to do
def to_plot(img):
	return np.rollaxis(img, 0, 1).astype(np.unit8)

# plot an image
def plot(img):
    plt.imshow(to_plot(img))

# access image loading function
from tensorflow.contrib.keras.python.keras.preprocessing.image import load_img

# plot a number of image
def plots_idx(idx, titles=None):
    plots([load_img(valid_path + filenames[i]) for i in idx], titles=titles)
