

"""
### Shared vision model

This model re-uses the same image-processing module on two inputs, to classify whether two MNIST digits are the same digit or different digits.

"""

from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate
from tensorflow.contrib.keras.python.keras.models import Model

# First, define the vision modules
# digit_input = Input(shape=(1, 27, 27))
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# Then define the tell-digits-apart model
# digit_a = Input(shape=(1, 27, 27))
# digit_b = Input(shape=(1, 27, 27))
digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# The vision model will be shared, weights and all
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)
