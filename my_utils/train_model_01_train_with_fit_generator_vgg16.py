"""
1. train model with batch_generators
2. generator = train_batches; validation_data = val_batches
3. steps_per_epoch = num_batches per epoch for training set;
3. epochs = num_epochs;
4. validation_steps = num_batches per epoch for validation set;

???
1. callbacks = ?
2. class_weight = ?
3. max_q_size = ?
4. initial_epoch = ?
"""

# to get avaliable train_batches, val_batches
from prep_data_02_img_folder_2_iterators import train_batches, val_batches

# to get the newly created vgg16
from finetune_model_01_finetune_vgg16 import vgg16_ft

## check model summary
vgg16_ft.summary()


# train the new model for number of epochs
# internally, there are lots of callbacks and queues, no idea how they work
vgg16_ft.fit_generator(
					generator=train_batches,
					steps_per_epoch=1, # num_batches for training set
					epochs=2,
					verbose=2,
					callbacks=None,
					validation_data=val_batches,
					validation_steps=1, # num_batches for validation set
					class_weight=None,
					max_q_size=10, # no idea what it work here
					workers=1, # how many threads to use or cpu to use
					pickle_safe=False,
					initial_epoch=0) # no idea how it work


vgg16_ft_trained = vgg16_ft
