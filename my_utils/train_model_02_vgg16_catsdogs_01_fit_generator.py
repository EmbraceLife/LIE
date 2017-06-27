"""
Use cases 1:
how to train one sample at a time?
- create batch_iterator with batch_size 1
- fit_generator set steps_per_epoch as 1, num_epochs as 1

Inputs:
1. finetuned_model
2. parameters:
	- verbose
	- num_epochs, train_batch_iterator, num_steps_per_epoch,
	- validation_data(iterator, tuple of arrays), validation_step for iterator
	- callbacks
	- initial_epoch

Return:
1. a trained model

Easy args:
1. verbose=2, to speak up as training is on
2. generator = train_batches; validation_data = val_batches
3. steps_per_epoch = num_batches per epoch for training set;
3. epochs = num_epochs to train
4. validation_steps = num_batches per epoch for validation set;

Args to explore:
1. callbacks: callbacks to help save best model during training
	- ModelCheckpoint(filepath=best_model_in_training, save_best_only=True, mode='min')
	- how to use it independently?
4. initial_epoch = 0 # 1
	- change from 0 to 1: the first epoch is not appear to be trained, and directly jump to train on the second epoch.
	- does it mean the first epoch of first time training will always be the same first epoch of dataset when train it for a second time?
"""

# to get avaliable train_batches, val_batches
from prep_data_02_vgg16_catsdogs_03_img_folder_2_iterators import train_batches, val_batches

# to get the newly created vgg16
from build_model_02_vgg16_catsdogs_01_finetune_vgg16 import vgg16_ft

## check model summary
vgg16_ft.summary()

# always read doc
# doc vgg16_ft.fit_generator


vgg16_ft.fit_generator(
					generator=train_batches,
					steps_per_epoch=1, # num_batches for training set
					epochs=2,
					verbose=2,
					callbacks=None, # [ModelCheckpoint(filepath=best_model_in_training, save_best_only=True, mode='min')]
					validation_data=val_batches,
					validation_steps=1, # num_batches for validation set
					class_weight=None,
					max_q_size=10, # no idea what it work here
					workers=1, # how many threads to use or cpu to use
					pickle_safe=False,
					initial_epoch=1)


vgg16_ft_trained = vgg16_ft
