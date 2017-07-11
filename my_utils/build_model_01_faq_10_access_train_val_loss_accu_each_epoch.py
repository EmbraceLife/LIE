"""
history_record_losses_metrics_all_epochs

### How can I record the training / validation loss / accuracy at each epoch?

The `model.fit` method returns an `History` callback, which has a `history` attribute containing the lists of successive losses and other metrics.
"""


from tensorflow.contrib.keras.python.keras.layers import Dropout, BatchNormalization, Input, Dense
from tensorflow.contrib.keras.python.keras.models import Model
import numpy as np
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import losses

x = np.random.random((100,10))*2
y = np.random.randint(2, size=(100,1))

input_tensor = Input(shape=(10,))
bn_tensor = BatchNormalization()(input_tensor)
dp_tensor = Dropout(0.7)(bn_tensor)
final_tensor = Dense(1)(dp_tensor)

model = Model(input_tensor, final_tensor)
model.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])

from tensorflow.contrib.keras.python.keras import callbacks

from tensorflow.contrib.keras.python.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5) # patience=2

hist1=model.fit(x, y, validation_split=0.2, callbacks=[early_stopping], epochs=100)
print(hist1.history)

hist2=model.fit(x, y, validation_split=0.3, epochs=10)
# checkout all the losses and metrics
print(hist2.history)
