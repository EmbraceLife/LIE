

"""
### Residual connection on a convolution layer

For more information about residual networks, see [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).

"""



from tensorflow.contrib.keras.python.keras.layers import add, Input, Conv2D

# input tensor for a 3-channel 256x256 image
# x = Input(shape=(3, 256, 256)) will cause error
x = Input(shape=(256, 256, 3))
# due to keras.backend._config uses channels_last

# 3x3 conv with 3 output channels (same as input channels)
y = Conv2D(3, (3, 3), padding='same')(x)
# this returns x + y.
z = add([x, y])
