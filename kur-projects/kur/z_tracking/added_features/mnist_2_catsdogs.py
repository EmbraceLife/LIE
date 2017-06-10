
"""
How to convert a folder of images with labels into a huge numpy arrays and a vector of 0s and 1s?

0. keep gray color of all images
1. convert all images into numpy.arrays
2. make all images equal shape or size
3. convert labels into numpy.array of 0s or 1s

"""
##########################################
# How to convert images to numpy arrays
from scipy import misc
import glob

for image_path in glob.glob("/home/adam/*.png"):
    image = misc.imread(image_path)
    # print image.shape
    # print image.dtype

##########################################
# How to preserve only gray color
from skimage import io
img = io.imread('image.png', as_grey=True)

# or
from skimage import color
from skimage import io

img = color.rgb2gray(io.imread('image.png'))


##########################################
# How to convert numpy array to images
# this should be easily found in mnist examples

###########################################
# How to resize images to the same shape
# http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
