# https://www.youtube.com/watch?v=68OrRqH2B_s&list=PLXO45tsB95cKiBRXYqNNCw8AUo6tYen3l&index=16

################################
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
import os


# get dirpath, file_path from a directory
####################################################
dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"
epoch_idx = 1 # i from animation()
for dirpath, _, filenames in os.walk(dirpath):
	# find filename with keynames such as "images" and "1"
	for filename in filenames:
		if filename.find("images") > -1 and filename.find(str(epoch_idx))> -1:
			img_file = dirpath + '/' + filename
			set_trace()
	pass

# method 1: subplot2grid for neuralnet layers
##########################
plt.figure(figsize=(5, 10))

ax1 = plt.subplot2grid((5,1), (0,0), rowspan=1, colspan=1)
dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"
epoch_idx = 1 # i from animation()
img_file=None
for dirpath, _, filenames in os.walk(dirpath):
	# find filename with keynames such as "images" and "1"
	for filename in filenames:
		if filename.find("images") > -1 and filename.find(str(epoch_idx))> -1:
			img_file = dirpath + '/' + filename

img=mpimg.imread(img_file)
# img_title = img_file[img_file.find(".png")
ax1.set_title("epoch1")
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
