

# to get avaliable train_batches, val_batches
from vgg16_02_iterator_from_directory import train_batches, val_batches

# to get the newly created vgg16
from vgg16_04_finetune import vgg16

## check model summary
vgg16.summary()


# train the new model for number of epochs
vgg16.fit_generator(
					generator=train_batches,
					steps_per_epoch=1,
					epochs=2,
					verbose=2,
					callbacks=None,
					validation_data=val_batches,
					validation_steps=1,
					class_weight=None,
					max_q_size=10,
					workers=1,
					pickle_safe=False,
					initial_epoch=0)

trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
vgg16.save(trained_model_path+'/vgg16_2class.h5')
#Notice we are passing in the validation dataset to the fit() method
#For each epoch we test our model against the validation set
# latest_weights_filename = None
# for epoch in range(no_of_epochs):
#     print "Running epoch: %d" % epoch
#     vgg.fit(batches, val_batches, nb_epoch=1)
#     latest_weights_filename = 'ft%d.h5' % epoch
#     vgg.model.save_weights(results_path+latest_weights_filename)
# print "Completed %s fit operations" % no_of_epochs
