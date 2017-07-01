"""
Uses:
1. build a simple inception model: multiple inputs, multiple towers
2. Residual connection on a convolution layer
3. a shared vision model
4. Visual question answering model
5. Video question answering model

### Inception module

For more information about the Inception architecture, see [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842).
"""

from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D, Input, concatenate

input_img = Input(shape=(3, 256, 256))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = concatenate([tower_1, tower_2, tower_3], axis=1)
