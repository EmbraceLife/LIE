# https://www.youtube.com/watch?v=68OrRqH2B_s&list=PLXO45tsB95cKiBRXYqNNCw8AUo6tYen3l&index=16

################################
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib import animation
from matplotlib.widgets import Slider
import numpy as np
import os

####################################################
####################################################
# experimental codes
####################################################
####################################################
"""
# get dirpath, file_path from a directory
####################################################
dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"
epoch_idx = 1 # i from animation()
for dirpath, _, filenames in os.walk(dirpath):
	# find filename with keynames such as "images" and "1"
	for filename in filenames:
		if filename.find("images") > -1 and filename.find(str(epoch_idx))> -1:
			img_file = dirpath + '/' + filename

	pass

# use subplots to put 5 different groups of images together
##########################

# dirpath is a global variable from above
dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"
# make an ordered list of layers to plot
layer_order = [['images'], ['convolution_0'], ['conv_layer1'], ['convolution_1'], ['conv_layer2']]

# define how many rows of subplots we need
num_cols = len(layer_order)

# i from animation(), start from 0, repetition
for i in range(10):
	epoch_idx = i
	# empty file name
	img_file = None
	# move it inside here, to ensure each epoch plot the same size figure
	plt.figure(figsize=(5, 10))
	# plot layer one by one
	for col in range(num_cols):

		# access all filenames in the dirpath
		for dirpath, _, filenames in os.walk(dirpath):
			# looping through every filename
			for filename in filenames:

				# search all filenames with a row's keywords
				# "epoch"+str(i+1): make sure all layers in one figure share the same epoch
				if filename.find(layer_order[col][0]) > -1 and filename.find(str(i+1)+".png")> -1:
					img_file = dirpath + '/' + filename
					img=mpimg.imread(img_file)

					ax1 = plt.subplot2grid((num_cols,1), (col,0), rowspan=1, colspan=1)
					ax1.set_title(layer_order[col][0] + "_epoch_" + str(i+1))
					ax1.imshow(img)

	# all subplots are stored inside plt
	me = plt
	me.show()
	set_trace()
"""


####################################################
# animation the above plots ##########################
####################################################

def animate(i):
	# dirpath is a global variable from above
	dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"

	# make an ordered list of layers to plot
	layer_order = [['images'], ['convolution_0'], ['conv_layer1'], ['convolution_1'], ['conv_layer2']]

	# define how many rows of subplots we need
	num_cols = len(layer_order)

	# i from animation(), start from 0, repetition

	# empty file name
	img_file = None

	# plt.figure(figsize=(5, 10))
	# plot layer one by one
	for col in range(num_cols):

		# access all filenames in the dirpath
		for dirpath, _, filenames in os.walk(dirpath):
			# looping through every filename
			for filename in filenames:

				# search all filenames with a row's keywords
				# "epoch"+str(i+1): make sure all layers in one figure share the same epoch
				if filename.find(layer_order[col][0]) > -1 and filename.find(str(i+1)+".png")> -1:
					img_file = dirpath + '/' + filename
					img=mpimg.imread(img_file)

					ax = plt.subplot2grid((3, num_cols), (0, col), rowspan=3, colspan=1)
					ax.set_title(layer_order[col][0] + "_epoch_" + str(i+1))
					ax.imshow(img)
	# plt.show() can plot all subplots, meaning all subplots are stored inside plt
	return plt


def init():
	# dirpath is a global variable from above
	dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"

	# make an ordered list of layers to plot
	layer_order = [['images'], ['convolution_0'], ['conv_layer1'], ['convolution_1'], ['conv_layer2']]

	# define how many rows of subplots we need
	num_cols = len(layer_order)

	# init() only plot on the first epoch
	epoch_idx = 1
	# empty file name
	img_file = None

	# plt.figure(figsize=(5, 10))
	# plot layer one by one
	for col in range(num_cols):

		# access all filenames in the dirpath
		for dirpath, _, filenames in os.walk(dirpath):
			# looping through every filename
			for filename in filenames:

				# search all filenames with a col's keywords
				# "epoch"+str(i+1): make sure all layers in one figure share the same epoch
				if filename.find(layer_order[col][0]) > -1 and filename.find(str(epoch_idx)+".png")> -1:
					img_file = dirpath + '/' + filename
					img=mpimg.imread(img_file)

					# set num_rows and num_cols 
					ax = plt.subplot2grid((3, num_cols), (0, col), rowspan=3, colspan=1)
					ax.autoscale(True)
					ax.set_title(layer_order[col][0] + "_epoch_" + str(epoch_idx))
					ax.imshow(img)

	# plt.subplots_adjust(left=0.25, bottom=0.25)
	axframe = plt.axes([0, 0, 0.5, 0.3])
	sframe = Slider(axframe, 'Frame', 0, 99, valinit=0,valfmt='%d')
	# plt.show() can plot all subplots, meaning all subplots are stored inside plt
	return plt


# call the animator.  blit=True means only re-draw the parts that have changed.
# blit=True dose not work on Mac, set blit=False
# interval= update frequency
fig = plt.figure(figsize=(20, 3))

ani = animation.FuncAnimation(fig=fig, func=animate, frames=10000, init_func=init,
                              interval=5000, blit=False)

plt.show()


set_trace()


# method 1: subplot2grid
##########################
plt.figure(5, 10)


ax1 = plt.subplot2grid((5,1), (0,0), rowspan=1, colspan=1)
img=mpimg.imread('/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights/images_epoch_1.png')
ax1.imshow(img)

ax2 = plt.subplot2grid((5,1), (1,0), rowspan=1, colspan=1)
img=mpimg.imread('/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights/convolution_0+weight.kur_epoch_1.png')
ax2.imshow(img)

ax3 = plt.subplot2grid((5,1), (2,0), rowspan=1, colspan=1)
img=mpimg.imread('/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights/conv_layer1_epoch_1.png')
ax3.imshow(img)

ax4 = plt.subplot2grid((5,1), (3,0), rowspan=1, colspan=1)
img=mpimg.imread('/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights/convolution_1+weight.kur_epoch_1.png')
ax4.imshow(img)

ax5 = plt.subplot2grid((5,1), (4,0), rowspan=1, colspan=1)
img=mpimg.imread('/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights/conv_layer2_epoch_1.png')
ax5.imshow(img)
plt.show()


# method 1: subplot2grid
##########################
plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)  # stands for axes
ax1.plot([1, 2], [1, 2])
ax1.set_title('ax1_title')
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax4.scatter([1, 2], [2, 2])
ax4.set_xlabel('ax4_x')
ax4.set_ylabel('ax4_y')
ax5 = plt.subplot2grid((3, 3), (2, 1))

# method 2: gridspec
#########################
plt.figure()
gs = gridspec.GridSpec(3, 3)
# use index from 0
ax6 = plt.subplot(gs[0, :])
ax7 = plt.subplot(gs[1, :2])
ax8 = plt.subplot(gs[1:, 2])
ax9 = plt.subplot(gs[-1, 0])
ax10 = plt.subplot(gs[-1, -2])

# method 3: easy to define structure
####################################
f, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, sharex=True, sharey=True)
ax11.scatter([1,2], [1,2])

plt.tight_layout()
plt.show()
