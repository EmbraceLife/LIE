"""
1. find file name and path for pre-trained model, and previous_epochs
2. load previous_epochs
3. load pre_trained model
4. train the model for a number of epochs, meanwhile save weights, model whenever you find appropriate
5. save total number of epochs trained as previous_epochs
"""


from prep_data_02_vgg16_catsdogs_03_img_folder_2_iterators import train_batches, val_batches
from tensorflow.contrib.keras.python.keras.models import load_model
import numpy as np
import os

trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"

# load log info: previous number of epochs
if os.path.isfile(trained_model_path+"/previous_epochs.npy"):
	previous_epochs = np.load(trained_model_path+"/previous_epochs.npy")[0]
	# select the latest trained model filename
	vgg16_again = load_model(trained_model_path+'/train_vgg16_again_model_5.h5')
else:
	print("no previous training epochs available, start from 0")
	previous_epochs = 0
	vgg16_again = load_model(trained_model_path+'/vgg16_ft_trained_model.h5')


## check model summary
vgg16_again.summary()

## train 2 epochs and save weights every 2 epochs, save model at the end of training
num_epochs=2

for epoch in range(num_epochs):
	print("Index iteration: %d" % epoch)
	vgg16_again.fit_generator(
						generator=train_batches,
						steps_per_epoch=1,
						epochs=1,
						verbose=2,
						callbacks=None,
						validation_data=val_batches,
						validation_steps=1,
						class_weight=None,
						max_q_size=10,
						workers=1,
						pickle_safe=False,
						initial_epoch=0)

	# save trained weights every 2 epochs
	if (epoch+1) % 2 == 0:

		print("save weights every 2 epochs")
		vgg16_again.save_weights(trained_model_path+'/train_vgg16_again_weights_%s.h5' % (epoch+previous_epochs))

	# save the model at the end of last epoch
	if epoch == num_epochs - 1:
		vgg16_again.save(trained_model_path+'/train_vgg16_again_model_%s.h5' % (epoch+previous_epochs))

# keep track of number of epochs trained
# the value be saved must be array alike, list, array etc
np.save(trained_model_path+"/previous_epochs.npy", [num_epochs+previous_epochs])

print("Trained %d epochs this time, and in total %d epochs" % (num_epochs, num_epochs+previous_epochs))
