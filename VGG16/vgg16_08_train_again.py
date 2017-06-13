from vgg16_iterator_from_directory import train_batches, val_batches

import tensorflow as tf
load_model = tf.contrib.keras.models.load_model
# load re-trained vgg16 model to test
trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
vgg16 = load_model(trained_model_path+'/tfkr_vgg16.h5') # already compiled


## check model summary
vgg16.summary()

## train 4 epochs and save model every 2 epochs
num_epochs=2
for epoch in range(num_epochs):

	# as verbose=2, so no print "Running epoch: %d" % epoch

	vgg16.fit_generator(
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

	if (epoch+1) % 2 == 0: # save model every 2 epochs
		# the first model is trained for one epoch, index 0
		# the second model is trained for another two epochs, index 2
		vgg16.save(trained_model_path+'/tfkr_vgg16_%s.h5' % (epoch+1))
		print("saved vgg16 at epoch_%s" % (epoch+1))
	# latest_weights_filename = 'ft%d.h5' % epoch
	# vgg.model.save_weights(results_path+latest_weights_filename)
print("Completed %d fit operations" % num_epochs)



############################################
# ## create fake data for training
# # input_1 (InputLayer)         (None, 224, 224, 3)
# fake_img = np.random.random((32*10, 224, 224, 3))
# fake_lab = np.random.random((32*10, 1000))
#
# lr = 0.001
# vgg16.compile(optimizer=Adam(lr=lr),
# 		loss='categorical_crossentropy', metrics=['accuracy'])
#
# vgg16.fit(x=fake_img, y=fake_lab, batch_size=32, epochs=1, verbose=2, callbacks=None, validation_split=0.1, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
