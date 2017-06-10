
# coding: utf-8

# # TensorFlow Tutorial #14
# # DeepDream
# 
# by [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org/)
# / [GitHub](https://github.com/Hvass-Labs/TensorFlow-Tutorials) / [Videos on YouTube](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ)

# ## Introduction
# 
# We have seen in the previous tutorials how to use the gradient of a neural network to generate images. Tutorials #11 and #12 showed how to generate adversarial noise using the gradient. Tutorial #13 showed how to use the gradient for generating images that caused specific features  inside the neural network to respond.
# 
# This tutorial uses a similar idea as the previous tutorials. But we will now use the gradient of the neural network to amplify patterns in the input image. This is commonly called the DeepDream algorithm, but there are actually many different variations of the technique.
# 
# This builds on the previous tutorials. You should be familiar with neural networks in general (e.g. Tutorial #01 and #02).

# ## Flowchart
# 
# This flowchart shows roughly the idea of the DeepDream algorithm. We use the Inception model which has many more layers than shown here. We use TensorFlow to automatically derive the gradient for a given layer in the network with respect to the input image. The gradient is then used to update the input image. This procedure is repeated a number of times until patterns have emerged and we are satisfied with the resulting image.
# 
# What happens is that the neural network sees small traces of the patterns in the image and we merely amplify the patterns using the gradient.
# 
# There are some details of the DeepDream algorithm not shown here, e.g. that the gradient is blurred, which has some advantages discussed further below. The gradient is also calculated in tiles so it can work on very high-resolution images without running out of computer memory.

# In[1]:

from IPython.display import Image, display
Image('images/14_deepdream_flowchart.png')


# ### Recursive Optimization
# 
# The Inception model was trained on images of fairly low resolution, presumably 200-300 pixels. So when we use images with much larger resolution, the DeepDream algorithm will create many small patterns in the image.
# 
# One solution is to downscale the input image to 200-300 pixels. But such a low resolution is pixelated and ugly.
# 
# Another solution is to repeatedly downscale the original image and run the DeepDream algorithm on each of the smaller versions of the image. This creates larger patterns in the image that are then refined at the higher resolution.
# 
# This flowchart shows roughly the idea. The algorithm is implemented recursively and supports any number of downscaling levels. The algorithm has several details not shown here, e.g. that the images are blurred slightly before being downscaled, and the original image is only blended somewhat with the DeepDream images to add some of the original detail back in.

# In[2]:

Image('images/14_deepdream_recursive_flowchart.png')


# ## Imports

# In[3]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math

# Image manipulation.
import PIL.Image
from scipy.ndimage.filters import gaussian_filter


# This was developed using Python 3.5.2 (Anaconda) and TensorFlow version:

# In[4]:

tf.__version__


# ## Inception Model

# Several of the previous tutorials used the so-called Inception model v3. In this tutorial we will use another variant of the Inception model. It is unclear exactly which variant it is, because the Google developers have (as usual) neglected to document their work. We will therefore refer to it as the "Inception 5h" model because that is the name of the zip-file, although it actually appears to be a simpler and earlier version of the Inception model.
# 
# The Inception 5h model is used because it is easier to work with: It takes input images of any size, and it seems to create prettier pictures than the Inception v3 model (see Tutorial #13).

# In[5]:

import inception5h


# The Inception 5h model is downloaded from the internet. This is the default directory where you want to save the data-files. The directory will be created if it does not exist.

# In[6]:

# inception.data_dir = 'inception/5h/'


# Download the data for the Inception model if it doesn't already exist in the directory. It is 50 MB.

# In[7]:

inception5h.maybe_download()


# Load the Inception model so it is ready to be used.

# In[8]:

model = inception5h.Inception5h()


# The Inception 5h model has many layers that can be used for DeepDreaming. We have made a list of the 12 most commonly used layers for easy reference.

# In[9]:

len(model.layer_tensors)


# ## Helper-functions for image manipulation

# This function loads an image and returns it as a numpy array of floating-points.

# In[10]:

def load_image(filename):
    image = PIL.Image.open(filename)

    return np.float32(image)


# Save an image as a jpeg-file. The image is given as a numpy array with pixel-values between 0 and 255.

# In[11]:

def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    
    # Convert to bytes.
    image = image.astype(np.uint8)
    
    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')


# This function plots an image. Using matplotlib gives low-resolution images. Using PIL gives pretty pictures.

# In[12]:

def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.
    
    if False:
        # Convert the pixel-values to the range between 0.0 and 1.0
        image = np.clip(image/255.0, 0.0, 1.0)
        
        # Plot using matplotlib.
        plt.imshow(image, interpolation='lanczos')
        plt.show()
    else:
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)
        
        # Convert pixels to bytes.
        image = image.astype(np.uint8)

        # Convert to a PIL-image and display it.
        display(PIL.Image.fromarray(image))


# Normalize an image so its values are between 0.0 and 1.0. This is useful for plotting the gradient.

# In[13]:

def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)
    
    return x_norm


# This function plots the gradient after normalizing it.

# In[14]:

def plot_gradient(gradient):
    # Normalize the gradient so it is between 0.0 and 1.0
    gradient_normalized = normalize_image(gradient)
    
    # Plot the normalized gradient.
    plt.imshow(gradient_normalized, interpolation='bilinear')
    plt.show()


# This function resizes an image. It can take a size-argument where you give it the exact pixel-size you want the image to be e.g. (100, 200). Or it can take a factor-argument where you give it the rescaling-factor you want to use e.g. 0.5 for halving the size of the image in each dimension.
# 
# This is implemented using PIL which is a bit lengthy because we are working on numpy arrays where the pixels are floating-point values. This is not supported by PIL so the image must be converted to 8-bit bytes while ensuring the pixel-values are within the proper limits. Then the image is resized and converted back to floating-point values.

# In[15]:

def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor
        
        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2.
        size = size[0:2]
    
    # The height and width is reversed in numpy vs. PIL.
    size = tuple(reversed(size))

    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)
    
    # Convert the pixels to 8-bit bytes.
    img = img.astype(np.uint8)
    
    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)
    
    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)
    
    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)

    return img_resized


# ## DeepDream Algorithm

# ### Gradient

# The following helper-functions calculate the gradient of an input image for use in the DeepDream algorithm. The Inception 5h model can accept images of any size, but very large images may use many giga-bytes of RAM. In order to keep the RAM-usage low we will split the input image into smaller tiles and calculate the gradient for each of the tiles. 
# 
# However, this may result in visible lines in the final images produced by the DeepDream algorithm. We therefore choose the tiles randomly so the locations of the tiles are always different. This makes the seams between the tiles invisible in the final DeepDream image.

# This is a helper-function for determining an appropriate tile-size. The desired tile-size is e.g. 400x400 pixels, but the actual tile-size will depend on the image-dimensions.

# In[16]:

def get_tile_size(num_pixels, tile_size=400):
    """
    num_pixels is the number of pixels in a dimension of the image.
    tile_size is the desired tile-size.
    """

    # How many times can we repeat a tile of the desired size.
    num_tiles = int(round(num_pixels / tile_size))
    
    # Ensure that there is at least 1 tile.
    num_tiles = max(1, num_tiles)
    
    # The actual tile-size.
    actual_tile_size = math.ceil(num_pixels / num_tiles)
    
    return actual_tile_size


# This helper-function computes the gradient for an input image. The image is split into tiles and the gradient is calculated for each tile. The tiles are chosen randomly to avoid visible seams / lines in the final DeepDream image.

# In[17]:

def tiled_gradient(gradient, image, tile_size=400):
    # Allocate an array for the gradient of the entire image.
    grad = np.zeros_like(image)

    # Number of pixels for the x- and y-axes.
    x_max, y_max, _ = image.shape

    # Tile-size for the x-axis.
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    # 1/4 of the tile-size.
    x_tile_size4 = x_tile_size // 4

    # Tile-size for the y-axis.
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    # 1/4 of the tile-size
    y_tile_size4 = y_tile_size // 4

    # Random start-position for the tiles on the x-axis.
    # The random value is between -3/4 and -1/4 of the tile-size.
    # This is so the border-tiles are at least 1/4 of the tile-size,
    # otherwise the tiles may be too small which creates noisy gradients.
    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        # End-position for the current tile.
        x_end = x_start + x_tile_size
        
        # Ensure the tile's start- and end-positions are valid.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        # Random start-position for the tiles on the y-axis.
        # The random value is between -3/4 and -1/4 of the tile-size.
        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            # End-position for the current tile.
            y_end = y_start + y_tile_size

            # Ensure the tile's start- and end-positions are valid.
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            # Get the image-tile.
            img_tile = image[x_start_lim:x_end_lim,
                             y_start_lim:y_end_lim, :]

            # Create a feed-dict with the image-tile.
            feed_dict = model.create_feed_dict(image=img_tile)

            # Use TensorFlow to calculate the gradient-value.
            g = session.run(gradient, feed_dict=feed_dict)

            # Normalize the gradient for the tile. This is
            # necessary because the tiles may have very different
            # values. Normalizing gives a more coherent gradient.
            g /= (np.std(g) + 1e-8)

            # Store the tile's gradient at the appropriate location.
            grad[x_start_lim:x_end_lim,
                 y_start_lim:y_end_lim, :] = g
            
            # Advance the start-position for the y-axis.
            y_start = y_end

        # Advance the start-position for the x-axis.
        x_start = x_end

    return grad


# ### Optimize Image
# 
# This function is the main optimization-loop for the DeepDream algorithm. It calculates the gradient of the given layer of the Inception model with regard to the input image. The gradient is then added to the input image so the mean value of the layer-tensor is increased. This process is repeated a number of times and amplifies whatever patterns the Inception model sees in the input image.

# In[18]:

def optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=False):
    """
    Use gradient ascent to optimize an image so it maximizes the
    mean value of the given layer_tensor.
    
    Parameters:
    layer_tensor: Reference to a tensor that will be maximized.
    image: Input image used as the starting point.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    show_gradient: Plot the gradient in each iteration.
    """

    # Copy the image so we don't overwrite the original image.
    img = image.copy()
    
    print("Image before:")
    plot_image(img)

    print("Processing image: ", end="")

    # Use TensorFlow to get the mathematical function for the
    # gradient of the given layer-tensor with regard to the
    # input image. This may cause TensorFlow to add the same
    # math-expressions to the graph each time this function is called.
    # It may use a lot of RAM and could be moved outside the function.
    gradient = model.get_gradient(layer_tensor)
    
    for i in range(num_iterations):
        # Calculate the value of the gradient.
        # This tells us how to change the image so as to
        # maximize the mean of the given layer-tensor.
        grad = tiled_gradient(gradient=gradient, image=img)
        
        # Blur the gradient with different amounts and add
        # them together. The blur amount is also increased
        # during the optimization. This was found to give
        # nice, smooth images. You can try and change the formulas.
        # The blur-amount is called sigma (0=no blur, 1=low blur, etc.)
        # We could call gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # which would not blur the colour-channel. This tends to
        # give psychadelic / pastel colours in the resulting images.
        # When the colour-channel is also blurred the colours of the
        # input image are mostly retained in the output image.
        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        # Scale the step-size according to the gradient-values.
        # This may not be necessary because the tiled-gradient
        # is already normalized.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image by following the gradient.
        img += grad * step_size_scaled

        if show_gradient:
            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))

            # Plot the gradient.
            plot_gradient(grad)
        else:
            # Otherwise show a little progress-indicator.
            print(". ", end="")

    print()
    print("Image after:")
    plot_image(img)
    
    return img


# ### Recursive Image Optimization
# 
# The Inception model was trained on fairly small images. The exact size is unclear but maybe 200-300 pixels in each dimension. If we use larger images such as 1920x1080 pixels then the `optimize_image()` function above will add many small patterns to the image.
# 
# This helper-function downscales the input image several times and runs each downscaled version through the `optimize_image()` function above. This results in larger patterns in the final image. It also speeds up the computation.

# In[19]:

def recursive_optimize(layer_tensor, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=10, step_size=3.0,
                       tile_size=400):
    """
    Recursively blur and downscale the input image.
    Each downscaled image is run through the optimize_image()
    function to amplify the patterns that the Inception model sees.

    Parameters:
    image: Input image used as the starting point.
    rescale_factor: Downscaling factor for the image.
    num_repeats: Number of times to downscale the image.
    blend: Factor for blending the original and processed images.

    Parameters passed to optimize_image():
    layer_tensor: Reference to a tensor that will be maximized.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    """

    # Do a recursive step?
    if num_repeats>0:
        # Blur the input image to prevent artifacts when downscaling.
        # The blur amount is controlled by sigma. Note that the
        # colour-channel is not blurred as it would make the image gray.
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

        # Downscale the image.
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)
            
        # Recursive call to this function.
        # Subtract one from num_repeats and use the downscaled image.
        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats-1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)
        
        # Upscale the resulting image back to its original size.
        img_upscaled = resize_image(image=img_result, size=image.shape)

        # Blend the original and processed images.
        image = blend * image + (1.0 - blend) * img_upscaled

    print("Recursive level:", num_repeats)

    # Process the image using the DeepDream algorithm.
    img_result = optimize_image(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size)
    
    return img_result


# ## TensorFlow Session

# We need a TensorFlow session to execute the graph. This is an interactive session so we can continue adding gradient functions to the computational graph.

# In[20]:

session = tf.InteractiveSession(graph=model.graph)


# ## Hulk

# In the first example we have an image of The Hulk. Note how the colours of the original image are mostly kept in the DeepDream images. This is because the gradient is blurred in its colour-channels so it becomes somewhat gray-scale and mainly changes the shape of the image and not so much its colour.

# In[21]:

image = load_image(filename='images/hulk.jpg')
plot_image(image)


# First we need a reference to the tensor inside the Inception model which we will maximize in the DeepDream optimization algorithm. In this case we select the entire 3rd layer of the Inception model (layer index 2). It has 192 channels and we will try and maximize the average value across all these channels.

# In[22]:

layer_tensor = model.layer_tensors[2]
layer_tensor


# Now run the DeepDream optimization algorithm for 10 iterations with a step-size of 6.0, which is twice as high as in the recursive optimizations below. We also show the gradient for each iteration and you should note the visible artifacts in the seams between the tiles.

# In[23]:

img_result = optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=6.0, tile_size=400,
                   show_gradient=True)


# You can save the DeepDream image if you like.

# In[24]:

# save_image(img_result, filename='deepdream_hulk.jpg')


# Now run the DeepDream algorithm recursively. We perform 5 recursive steps (`num_repeats+1`) where the image is blurred and downscaled in each step and then the DeepDream algorithm is used on the downscaled image. The resulting DeepDream image is then blended with the original image in each step to add a little of the detail from the original image. This is repeated a number of times.
# 
# Note how the DeepDream patterns are now larger. This is because the patterns were first created on the low-resolution image and then refined on the higher-resolution images.

# In[25]:

img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)


# Now we will maximize a higher layer in the Inception model. In this case it is layer number 7 (index 6). This layer recognizes more complex shapes in the input image and the DeepDream algorithm will therefore produce a more complex image. This layer appears to be recognizing dog-faces and fur which the DeepDream algorithm has therefore added to the image.
# 
# Note again that the colours of the input image are mostly retained as opposed to other variants of the DeepDream algorithm which create more pastel-like colours. This is because we are also smoothing the gradient in the colour-channels so it becomes somewhat gray-scale and hence does not change the colours of the input image so much.

# In[ ]:

layer_tensor = model.layer_tensors[6]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)


# This is an example of maximizing only a subset of a layer's feature-channels using the DeepDream algorithm. In this case it is the layer with index 7 and only its first 3 feature-channels that are maximized.

# In[ ]:

layer_tensor = model.layer_tensors[7][:,:,:,0:3]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)


# This example shows the result of maximizing the first feature-channel of the final layer in the Inception model. It is unclear what patterns this layer and feature-channel might be recognizing in the input image.

# In[ ]:

layer_tensor = model.layer_tensors[11][:,:,:,0]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)


# ## Giger

# In[ ]:

image = load_image(filename='images/giger.jpg')
plot_image(image)


# In[ ]:

layer_tensor = model.layer_tensors[3]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)


# In[ ]:

layer_tensor = model.layer_tensors[5]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)


# ## Escher

# In[ ]:

image = load_image(filename='images/escher_planefilling2.jpg')
plot_image(image)


# In[ ]:

layer_tensor = model.layer_tensors[6]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)


# ## Close TensorFlow Session
# 
# We are now done using TensorFlow, so we close the session to release its resources.

# In[ ]:

# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
# session.close()


# ## Conclusion
# 
# This tutorial showed how to use the gradient of a neural network to amplify patterns in an image. The output images appeared to have been redrawn with abstract or animal-like patterns.
# 
# There are numerous variations of the technique that result in different output images. You are encouraged to experiment by changing the parameters and algorithms above.

# ## Exercises
# 
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
# 
# You may want to backup this Notebook and the other files before making any changes.
# 
# Exercises:
# 
# * Try using your own images.
# * Try experimenting with the parameters for `optimize_image()` and `recursive_optimize()` to see how it affects the result.
# * Try and subtract the gradient in `optimize_image()` instead of adding it. What happens?
# * Plot the gradients when you run `optimize_image()`. Do you see any artifacts? What do you think is the cause? Does it matter? Can you find a way to remove them?
# * Try using random noise as the input image. This is similar to Tutorial #13 for visually analyzing the features. Are the generated images better in this tutorial? Why?
# * Try and remove `tf.square()` inside `Inception5h.get_gradient()` in the file `inception5h.py`. What happens to the DeepDream images? Why is that?
# * Can you move the gradient outside of `optimize_image()` to save memory?
# * Can you make the program run faster? An idea would be to implement the gaussian blur and resizing directly in TensorFlow.
# * Make a DeepDream movie by repeatedly calling `optimize_image()` and zooming a little on the image.
# * Process a movie frame-by-frame. You may need to use some kind of stabilization across the individual frames.
# * Explain to a friend how the program works.

# ## License (MIT)
# 
# Copyright (c) 2016 by [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org/)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
