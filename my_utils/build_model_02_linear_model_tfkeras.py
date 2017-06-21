from tensorflow.contrib.keras.python.keras.models import Sequential

model = Sequential([
        BatchNormalization(axis=1, input_shape=(3,224,224)),
        Flatten(),
        Dense(10, activation='softmax')
    ])


model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches,
                 nb_val_samples=val_batches.nb_sample)


model.summary()
