
"""
early_stopping

### Early stopping training when the validation loss isn't decreasing anymore

callbacks: EarlyStopping

- monitor: 'val_loss' and possibly others too, but no idea what are they
- patince: wait for num of losses increase before stop
- mode: auto, min or max
- min_delta: measure what is a real improvement

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
model.compile(optimizer='SGD', loss='mse')

from tensorflow.contrib.keras.python.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5) # patience=2
"""
  '  def __init__(self,\n',
  "               monitor='val_loss',\n",
  '               min_delta=0,\n',
  '               patience=0,\n',
  '               verbose=0,\n',
  "               mode='auto'):\n",
  '    super(EarlyStopping, self).__init__()\n',

('Stop training when a monitored quantity has stopped improving.\n'
 '\n'
 'Arguments:\n'
 '    monitor: quantity to be monitored.\n'
 '    min_delta: minimum change in the monitored quantity\n'
 '        to qualify as an improvement, i.e. an absolute\n'
 '        change of less than min_delta, will count as no\n'
 '        improvement.\n'
 '    patience: number of epochs with no improvement\n'
 '        after which training will be stopped.\n'
 '    verbose: verbosity mode.\n'
 '    mode: one of {auto, min, max}. In `min` mode,\n'
 '        training will stop when the quantity\n'
 '        monitored has stopped decreasing; in `max`\n'
 '        mode it will stop when the quantity\n'
 '        monitored has stopped increasing; in `auto`\n'
 '        mode, the direction is automatically inferred\n'
 '        from the name of the monitored quantity.')
"""

# model.fit(x, y, validation_split=0.2, callbacks=[early_stopping], epochs=10)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping], epochs=100)
