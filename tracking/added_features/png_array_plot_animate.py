################################
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
"""
How to convert a folder of images with labels into a huge numpy arrays and a vector of 0s and 1s?

0. keep gray color of all images
1. convert all images into numpy.arrays
2. make all images equal shape or size
3. convert labels into numpy.array of 0s or 1s

"""
##########################################
# image (png) to numpy array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# https://matplotlib.org/users/image_tutorial.html


##########################################
# image (png) to numpy array
img=mpimg.imread('/Users/Natsume/Downloads/morvan_tutorials/片头.png')
img.shape
pprint(img.shape)


##########################################
# numpy array to plotting in python
imgplot = plt.imshow(img)
# pprint(imgplot.__dict__.keys())
# plt.show()


##########################################
# animate plotting images from png files

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig, ax = plt.subplots()

img=mpimg.imread('/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights/conv_layer1_epoch_1.png')
# imgplot = plt.imshow(img)
imgplot = ax.imshow(img)

# plt.show() # plot it directly


def animate(i):
	# update a sine line
    # line.set_ydata(np.sin(x + i/10.0))  # update the data
    # return line,
    img=mpimg.imread('/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights/conv_layer1_epoch_{}.png'.format(i+2))
	# imgplot = plt.imshow(img)
    imgplot = ax.imshow(img)
    return imgplot

# Init only required for blitting to give a clean slate.
def init():
    img=mpimg.imread('/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights/conv_layer1_epoch_1.png')
	# imgplot = plt.imshow(img)
    imgplot = ax.imshow(img)
    return imgplot

# call the animator.  blit=True means only re-draw the parts that have changed.
# blit=True dose not work on Mac, set blit=False
# interval= update frequency
ani = animation.FuncAnimation(fig=fig, func=animate, frames=1000, init_func=init,
                              interval=5000, blit=False)

# set_trace()

plt.show()
