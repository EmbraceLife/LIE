
# coding: utf-8

# # TensorFlow Tutorial #07
# # Inception Model
# 
# by [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org/)
# / [GitHub](https://github.com/Hvass-Labs/TensorFlow-Tutorials) / [Videos on YouTube](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ)

# ## Introduction
# 
# This tutorial shows how to use a pre-trained Deep Neural Network called Inception v3 for image classification.
# 
# The Inception v3 model takes weeks to train on a monster computer with 8 Tesla K40 GPUs and probably costing $30,000 so it is impossible to train it on an ordinary PC. We will instead download the pre-trained Inception model and use it to classify images.
# 
# The Inception v3 model has nearly 25 million parameters and uses 5 billion multiply-add operations for classifying a single image. On a modern PC without a GPU this can be done in a fraction of a second per image.
# 
# This tutorial hides the TensorFlow code so it may not require much experience with TensorFlow, although a basic understanding of TensorFlow from the previous tutorials might be helpful, especially if you want to study the implementation details in the `inception.py` file.

# ## Flowchart

# The following chart shows how the data flows in the Inception v3 model, which is a Convolutional Neural Network with many layers and a complicated structure. The [research paper](http://arxiv.org/pdf/1512.00567v3.pdf) gives more details on how the Inception model is constructed and why it is designed that way. But the authors admit that they don't fully understand why it works.
# 
# Note that the Inception model has two softmax-outputs. One is used during training of the neural network and the other is used for classifying images after training has finished, also known as inference.
# 
# [Newer models](https://research.googleblog.com/2016/08/improving-inception-and-image.html) became available just last week, which are even more complicated than Inception v3 and achieve somewhat better classification accuracy.

# In[1]:

from IPython.display import Image, display
Image('images/07_inception_flowchart.png')


# ## Imports

# In[2]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# Functions and classes for loading and using the Inception model.
import inception


# This was developed using Python 3.5.2 (Anaconda) and TensorFlow version:

# In[3]:

tf.__version__


# ## Download the Inception Model

# The Inception model is downloaded from the internet. This is the default directory where you want to save the data-files. The directory will be created if it does not exist.

# In[4]:

# inception.data_dir = 'inception/'


# Download the data for the Inception model if it doesn't already exist in the directory. It is 85 MB.

# In[5]:

inception.maybe_download()


# ## Load the Inception Model

# Load the Inception model so it is ready for classifying images.
# 
# Note the deprecation warning, which might cause the program to fail in the future.

# In[6]:

model = inception.Inception()


# ## Helper-function for classifying and plotting images

# This is a simple wrapper-function for displaying the image, then classifying it using the Inception model and finally printing the classification scores.

# In[7]:

def classify(image_path):
    # Display the image.
    display(Image(image_path))

    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)

    # Print the scores and names for the top-10 predictions.
    model.print_scores(pred=pred, k=10, only_first_name=True)    


# ## Panda

# This image of a panda is included in the Inception data-file. The Inception model is quite confident that this image shows a panda, with a classification score of 89.23% and the next highest score being only 0.86% for an indri, which is another exotic animal.

# In[8]:

image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path)


# ## Interpretation of Classification Scores
# 
# The output of the Inception model is a so-called Softmax-function which was also used in the neural networks in the previous tutorials.
# 
# The softmax-outputs are sometimes called probabilities because they are between zero and one, and they also sum to one - just like probabilities. But they are actually not probabilities in the traditional sense of the word, because they do not come from repeated experiments.
# 
# It is perhaps better to call the output values of a neural network for classification scores or ranks, because they indicate how strongly the network believes that the input image is of each possible class.
# 
# In the above example with the image of a panda, the Inception model gave a very high score of 89.23% for the panda-class, and the scores for the remaining 999 possible classes were all below 1%. This means the Inception model was quite certain that the image showed a panda and the remaining scores below 1% should be regarded as noise. For example, the 10th highest score was 0.05% for a digital watch, but this is probably more due to the imprecise nature of neural networks rather than an indication that the image looked slightly like a digital watch.
# 
# Sometimes the Inception model is confused about which class an image belongs to, so none of the scores are really high. Examples of this are shown below.

# ## Parrot (Original Image)

# The Inception model is very confident (score 97.30%) that this image shows a kind of parrot called a macaw.

# In[9]:

classify(image_path="images/parrot.jpg")


# ## Parrot (Resized Image)

# The Inception model works on input images that are 299 x 299 pixels in size. The above image of a parrot is actually 320 pixels wide and 785 pixels high, so it is resized automatically by the Inception model.
# 
# We now want to see the image after it has been resized by the Inception model.
# 
# First we have a helper-function for getting the resized image from inside the Inception model.

# In[10]:

def plot_resized_image(image_path):
    # Get the resized image from the Inception model.
    resized_image = model.get_resized_image(image_path=image_path)

    # Plot the image.
    plt.imshow(resized_image, interpolation='nearest')
    
    # Ensure that the plot is shown.
    plt.show()


# Now plot the resized image of the parrot. This is the image that is actually input to the neural network of the Inception model. We can see that it has been squeezed so it is rectangular, and the resolution has been reduced so the image has become more pixelated and grainy.
# 
# In this case the image still clearly shows a parrot, but some images may become so distorted from this naive resizing that you may want to resize the images yourself before inputting them to the Inception model.

# In[11]:

plot_resized_image(image_path="images/parrot.jpg")


# ## Parrot (Cropped Image, Top)

# This image of the parrot has been cropped manually to 299 x 299 pixels and then input to the Inception model, which is still very confident (score 97.38%) that it shows a parrot (macaw).

# In[12]:

classify(image_path="images/parrot_cropped1.jpg")


# ## Parrot (Cropped Image, Middle)

# This is another crop of the parrot image, this time showing its body without the head or tail. The Inception model is still very confident (score 93.94%) that it shows a macaw parrot.

# In[13]:

classify(image_path="images/parrot_cropped2.jpg")


# ## Parrot (Cropped Image, Bottom)

# This image has been cropped so it only shows the tail of the parrot. Now the Inception model is quite confused and thinks the image might show a jacamar (score 26.11%) which is another exotic bird, or perhaps the image shows a grass-hopper (score 10.61%).
# 
# The Inception model also thinks the image might show a fountain-pen (score 2.00%). But this is a very low score and should be interpreted as unreliable noise.

# In[14]:

classify(image_path="images/parrot_cropped3.jpg")


# ## Parrot (Padded Image)

# The best way to input images to this Inception model, is to pad the image so it is rectangular and then resize the image to 299 x 299 pixels, like this example of the parrot which is classified correctly with a score of 96.78%.

# In[15]:

classify(image_path="images/parrot_padded.jpg")


# ## Elon Musk (299 x 299 pixels)

# This image shows the living legend and super-nerd-hero Elon Musk. But the Inception model is very confused about what the image shows, predicting that it maybe shows a sweatshirt (score 19.73%) or an abaya (score 16.82%). It also thinks the image might show a ping-pong ball (score 3.05%) or a baseball (score 1.86%). So the Inception model is confused and the classification scores are unreliable.

# In[16]:

classify(image_path="images/elon_musk.jpg")


# ## Elon Musk (100 x 100 pixels)

# If we instead use a 100 x 100 pixels image of Elon Musk, then the Inception model thinks it might show a sweatshirt (score 17.85%) or a cowboy boot (score 16.36%). So now the Inception model has somewhat different predictions but it is still very confused.

# In[17]:

classify(image_path="images/elon_musk_100x100.jpg")


# The Inception model automatically upscales this image from 100 x 100 to 299 x 299 pixels, which is shown here. Note how pixelated and grainy it really is, although a human can easily see that this is a picture of a man with crossed arms.

# In[18]:

plot_resized_image(image_path="images/elon_musk_100x100.jpg")


# ## Willy Wonka (Gene Wilder)

# This image shows the actor Gene Wilder portraying Willy Wonka in the 1971 version of the movie. The Inception model is very confident that the image shows a bow tie (score 97.22%), which is true but a human would probably say this image shows a person.
# 
# The reason might be that the Inception model was trained on images of people with bow-ties that were classified as a bow-tie rather than a person. So maybe the problem is that the class-name should be "person with bow-tie" instead of just "bow-tie".

# In[19]:

classify(image_path="images/willy_wonka_old.jpg")


# ## Willy Wonka (Johnny Depp)

# This image shows the actor Johnny Depp portraying Willy Wonka in the 2005 version of the movie. The Inception model thinks that this image shows "sunglasses" (score 31.48%) or "sunglass" (score 18.77%). Actually, the full name of the first class is "sunglasses, dark glasses, shades". For some reason the Inception model has been trained to recognize two very similar classes for sunglasses. Once again, it is correct that the image shows sunglasses, but a human would probably have said that this image shows a person.

# In[20]:

classify(image_path="images/willy_wonka_new.jpg")


# ## Close TensorFlow Session

# We are now done using TensorFlow, so we close the session to release its resources. Note that the TensorFlow-session is inside the model-object, so we close the session through that object.

# In[21]:

# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
# model.close()


# ## Conclusion
# 
# This tutorial showed how to use the pre-trained Inception v3 model. It takes several weeks for a monster-computer to train the Inception model, but we can just download the finished model from the internet and use it on a normal PC for classifying images.
# 
# Unfortunately, the Inception model appears to have problems recognizing people. This may be due to the training-set that was used. Newer versions of the Inception model have already been released, but they are probably also trained on the same data-set and may therefore also have problems recognizing people. Future models will hopefully be trained to recognize common objects such as people.
# 
# In this tutorial we have hidden the TensorFlow implementation in the `inception.py` file because it is a bit messy and we may want to re-use it in future tutorials. Hopefully the TensorFlow developers will standardize and simplify the API for loading these pre-trained models more easily, so that anyone can use a powerful image classifier with just a few lines of code.

# ## Exercises
# 
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
# 
# You may want to backup this Notebook and the other files before making any changes.
# 
# * Use your own images, or images you find on the internet.
# * Crop, resize and distort the images to see how it affects the classification accuracy.
# * Add print-statements to various places in the code to see what data is being passed around in the code. You can also run and debug the `inception.py` file directly.
# * Try and use one of the [newer models](https://research.googleblog.com/2016/08/improving-inception-and-image.html) that were just released. These are loaded in a different way than the Inception v3 model and may be more challenging to implement.
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
