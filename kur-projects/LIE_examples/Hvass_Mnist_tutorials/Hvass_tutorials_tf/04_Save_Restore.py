
# coding: utf-8

# # TensorFlow Tutorial #04
# # Save & Restore
# 
# by [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org/)
# / [GitHub](https://github.com/Hvass-Labs/TensorFlow-Tutorials) / [Videos on YouTube](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ)

# ## Introduction
# 
# This tutorial demonstrates how to save and restore the variables of a Neural Network. During optimization we save the variables of the neural network whenever its classification accuracy has improved on the validation-set. The optimization is aborted when there has been no improvement for 1000 iterations. We then reload the variables that performed best on the validation-set.
# 
# This strategy is called Early Stopping. It is used to avoid overfitting of the neural network. This occurs when the neural network is being trained for too long so it starts to learn the noise of the training-set, which causes the neural network to mis-classify new images.
# 
# Overfitting is not really a problem for the neural network used in this tutorial on the MNIST data-set for recognizing hand-written digits. But this tutorial demonstrates the idea of Early Stopping.
# 
# This builds on the previous tutorials, so you should have a basic understanding of TensorFlow and the add-on package Pretty Tensor. A lot of the source-code and text in this tutorial is similar to the previous tutorials and may be read quickly if you have recently read the previous tutorials.

# ## Flowchart

# The following chart shows roughly how the data flows in the Convolutional Neural Network that is implemented below. The network has two convolutional layers and two fully-connected layers, with the last layer being used for the final classification of the input images. See Tutorial #02 for a more detailed description of this network and convolution in general.

# In[1]:

from IPython.display import Image
Image('images/02_network_flowchart.png')


# ## Imports

# In[2]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt


# This was developed using Python 3.5.2 (Anaconda) and TensorFlow version:

# In[3]:

tf.__version__


# PrettyTensor version:

# In[4]:

pt.__version__


# ## Load Data

# The MNIST data-set is about 12 MB and will be downloaded automatically if it is not located in the given path.

# In[5]:

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


# The MNIST data-set has now been loaded and consists of 70,000 images and associated labels (i.e. classifications of the images). The data-set is split into 3 mutually exclusive sub-sets. We will only use the training and test-sets in this tutorial.

# In[6]:

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))


# The class-labels are One-Hot encoded, which means that each label is a vector with 10 elements, all of which are zero except for one element. The index of this one element is the class-number, that is, the digit shown in the associated image. We also need the class-numbers as integers for the test- and validation-sets, so we calculate them now.

# In[7]:

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)


# ## Data Dimensions

# The data dimensions are used in several places in the source-code below. They are defined once so we can use these variables instead of numbers throughout the source-code below.

# In[8]:

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10


# ### Helper-function for plotting images

# Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.

# In[9]:

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ### Plot a few images to see if data is correct

# In[10]:

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


# ## TensorFlow Graph
# 
# The entire purpose of TensorFlow is to have a so-called computational graph that can be executed much more efficiently than if the same calculations were to be performed directly in Python. TensorFlow can be more efficient than NumPy because TensorFlow knows the entire computation graph that must be executed, while NumPy only knows the computation of a single mathematical operation at a time.
# 
# TensorFlow can also automatically calculate the gradients that are needed to optimize the variables of the graph so as to make the model perform better. This is because the graph is a combination of simple mathematical expressions so the gradient of the entire graph can be calculated using the chain-rule for derivatives.
# 
# TensorFlow can also take advantage of multi-core CPUs as well as GPUs - and Google has even built special chips just for TensorFlow which are called TPUs (Tensor Processing Units) and are even faster than GPUs.
# 
# A TensorFlow graph consists of the following parts which will be detailed below:
# 
# * Placeholder variables used for inputting data to the graph.
# * Variables that are going to be optimized so as to make the convolutional network perform better.
# * The mathematical formulas for the convolutional network.
# * A loss measure that can be used to guide the optimization of the variables.
# * An optimization method which updates the variables.
# 
# In addition, the TensorFlow graph may also contain various debugging statements e.g. for logging data to be displayed using TensorBoard, which is not covered in this tutorial.

# ### Placeholder variables

# Placeholder variables serve as the input to the TensorFlow computational graph that we may change each time we execute the graph. We call this feeding the placeholder variables and it is demonstrated further below.
# 
# First we define the placeholder variable for the input images. This allows us to change the images that are input to the TensorFlow graph. This is a so-called tensor, which just means that it is a multi-dimensional array. The data-type is set to `float32` and the shape is set to `[None, img_size_flat]`, where `None` means that the tensor may hold an arbitrary number of images with each image being a vector of length `img_size_flat`.

# In[11]:

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


# The convolutional layers expect `x` to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead `[num_images, img_height, img_width, num_channels]`. Note that `img_height == img_width == img_size` and `num_images` can be inferred automatically by using -1 for the size of the first dimension. So the reshape operation is:

# In[12]:

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


# Next we have the placeholder variable for the true labels associated with the images that were input in the placeholder variable `x`. The shape of this placeholder variable is `[None, num_classes]` which means it may hold an arbitrary number of labels and each label is a vector of length `num_classes` which is 10 in this case.

# In[13]:

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')


# We could also have a placeholder variable for the class-number, but we will instead calculate it using argmax. Note that this is a TensorFlow operator so nothing is calculated at this point.

# In[14]:

y_true_cls = tf.argmax(y_true, dimension=1)


# ### Neural Network

# This section implements the Convolutional Neural Network using Pretty Tensor, which is much simpler than a direct implementation in TensorFlow, see Tutorial #03.
# 
# The basic idea is to wrap the input tensor `x_image` in a Pretty Tensor object which has helper-functions for adding new computational layers so as to create an entire neural network. Pretty Tensor takes care of the variable allocation, etc.

# In[15]:

x_pretty = pt.wrap(x_image)


# Now that we have wrapped the input image in a Pretty Tensor object, we can add the convolutional and fully-connected layers in just a few lines of source-code.
# 
# Note that `pt.defaults_scope(activation_fn=tf.nn.relu)` makes `activation_fn=tf.nn.relu` an argument for each of the layers constructed inside the `with`-block, so that Rectified Linear Units (ReLU) are used for each of these layers. The `defaults_scope` makes it easy to change arguments for all of the layers.

# In[16]:

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.        conv2d(kernel=5, depth=16, name='layer_conv1').        max_pool(kernel=2, stride=2).        conv2d(kernel=5, depth=36, name='layer_conv2').        max_pool(kernel=2, stride=2).        flatten().        fully_connected(size=128, name='layer_fc1').        softmax_classifier(num_classes=num_classes, labels=y_true)


# ### Getting the Weights

# Further below, we want to plot the weights of the neural network. When the network is constructed using Pretty Tensor, all the variables of the layers are created indirectly by Pretty Tensor. We therefore have to retrieve the variables from TensorFlow.
# 
# We used the names `layer_conv1` and `layer_conv2` for the two convolutional layers. These are also called variable scopes (not to be confused with `defaults_scope` as described above). Pretty Tensor automatically gives names to the variables it creates for each layer, so we can retrieve the weights for a layer using the layer's scope-name and the variable-name.
# 
# The implementation is somewhat awkward because we have to use the TensorFlow function `get_variable()` which was designed for another purpose; either creating a new variable or re-using an existing variable. The easiest thing is to make the following helper-function.

# In[17]:

def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


# Using this helper-function we can retrieve the variables. These are TensorFlow objects. In order to get the contents of the variables, you must do something like: `contents = session.run(weights_conv1)` as demonstrated further below.

# In[18]:

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')


# ### Optimization Method

# Pretty Tensor gave us the predicted class-label (`y_pred`) as well as a loss-measure that must be minimized, so as to improve the ability of the neural network to classify the input images.
# 
# It is unclear from the documentation for Pretty Tensor whether the loss-measure is cross-entropy or something else. But we now use the `AdamOptimizer` to minimize the loss.
# 
# Note that optimization is not performed at this point. In fact, nothing is calculated at all, we just add the optimizer-object to the TensorFlow graph for later execution.

# In[19]:

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)


# ### Performance Measures
# 
# We need a few more performance measures to display the progress to the user.
# 
# First we calculate the predicted class number from the output of the neural network `y_pred`, which is a vector with 10 elements. The class number is the index of the largest element.

# In[20]:

y_pred_cls = tf.argmax(y_pred, dimension=1)


# Then we create a vector of booleans telling us whether the predicted class equals the true class of each image.

# In[21]:

correct_prediction = tf.equal(y_pred_cls, y_true_cls)


# The classification accuracy is calculated by first type-casting the vector of booleans to floats, so that False becomes 0 and True becomes 1, and then taking the average of these numbers.

# In[22]:

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ### Saver
# 
# In order to save the variables of the neural network, we now create a so-called Saver-object which is used for storing and retrieving all the variables of the TensorFlow graph. Nothing is actually saved at this point, which will be done further below in the `optimize()`-function.

# In[23]:

saver = tf.train.Saver()


# The saved files are often called checkpoints because they may be written at regular intervals during optimization.
# 
# This is the directory used for saving and retrieving the data.

# In[24]:

save_dir = 'checkpoints/'


# Create the directory if it does not exist.

# In[25]:

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# This is the path for the checkpoint-file.

# In[26]:

save_path = os.path.join(save_dir, 'best_validation')


# ## TensorFlow Run

# ### Create TensorFlow session
# 
# Once the TensorFlow graph has been created, we have to create a TensorFlow session which is used to execute the graph.

# In[27]:

session = tf.Session()


# ### Initialize variables
# 
# The variables for `weights` and `biases` must be initialized before we start optimizing them. We make a simple wrapper-function for this, because we will call it again below.

# In[28]:

def init_variables():
    session.run(tf.global_variables_initializer())


# Execute the function now to initialize the variables.

# In[29]:

init_variables()


# ### Helper-function to perform optimization iterations

# There are 55,000 images in the training-set. It takes a long time to calculate the gradient of the model using all these images. We therefore only use a small batch of images in each iteration of the optimizer.
# 
# If your computer crashes or becomes very slow because you run out of RAM, then you may try and lower this number, but you may then need to perform more optimization iterations.

# In[30]:

train_batch_size = 64


# The classification accuracy for the validation-set will be calculated for every 100 iterations of the optimization function below. The optimization will be stopped if the validation accuracy has not been improved in 1000 iterations. We need a few variables to keep track of this.

# In[31]:

# Best validation accuracy seen so far.
best_validation_accuracy = 0.0

# Iteration-number for last improvement to validation accuracy.
last_improvement = 0

# Stop optimization if no improvement found in this many iterations.
require_improvement = 1000


# Function for performing a number of optimization iterations so as to gradually improve the variables of the network layers. In each iteration, a new batch of data is selected from the training-set and then TensorFlow executes the optimizer using those training samples.  The progress is printed every 100 iterations where the validation accuracy is also calculated and saved to a file if it is an improvement.

# In[32]:

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variables rather than local copies.
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        # Increase the total number of iterations performed.
        # It is easier to update it in each iteration because
        # we need this number several times in the following.
        total_iterations += 1

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations and after last iteration.
        if (total_iterations % 100 == 0) or (i == (num_iterations - 1)):

            # Calculate the accuracy on the training-batch.
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)

            # Calculate the accuracy on the validation-set.
            # The function returns 2 values but we only need the first.
            acc_validation, _ = validation_accuracy()

            # If validation accuracy is an improvement over best-known.
            if acc_validation > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = acc_validation
                
                # Set the iteration for the last improvement to current.
                last_improvement = total_iterations

                # Save all variables of the TensorFlow graph to file.
                saver.save(sess=session, save_path=save_path)

                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''
            
            # Status-message for printing.
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"

            # Print it.
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))

        # If no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")

            # Break out from the for-loop.
            break

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# ### Helper-function to plot example errors

# Function for plotting examples of images from the test-set that have been mis-classified.

# In[33]:

def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# ### Helper-function to plot confusion matrix

# In[34]:

def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ### Helper-functions for calculating classifications
# 
# This function calculates the predicted classes of images and also returns a boolean array whether the classification of each image is correct.
# 
# The calculation is done in batches because it might use too much RAM otherwise. If your computer crashes then you can try and lower the batch-size.

# In[35]:

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


# Calculate the predicted class for the test-set.

# In[36]:

def predict_cls_test():
    return predict_cls(images = data.test.images,
                       labels = data.test.labels,
                       cls_true = data.test.cls)


# Calculate the predicted class for the validation-set.

# In[37]:

def predict_cls_validation():
    return predict_cls(images = data.validation.images,
                       labels = data.validation.labels,
                       cls_true = data.validation.cls)


# ### Helper-functions for the classification accuracy
# 
# This function calculates the classification accuracy given a boolean array whether each image was correctly classified. E.g. `cls_accuracy([True, True, False, False, False]) = 2/5 = 0.4`

# In[38]:

def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum


# Calculate the classification accuracy on the validation-set.

# In[39]:

def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()
    
    # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)


# ### Helper-function for showing the performance

# Function for printing the classification accuracy on the test-set.
# 
# It takes a while to compute the classification for all the images in the test-set, that's why the results are re-used by calling the above functions directly from this function, so the classifications don't have to be recalculated by each function.

# In[40]:

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = cls_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# ### Helper-function for plotting convolutional weights

# In[41]:

def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ## Performance before any optimization
# 
# The accuracy on the test-set is very low because the model variables have only been initialized and not optimized at all, so it just classifies the images randomly.

# In[42]:

print_test_accuracy()


# The convolutional weights are random, but it can be difficult to see any difference from the optimized weights that are shown below. The mean and standard deviation is shown so we can see whether there is a difference.

# In[43]:

plot_conv_weights(weights=weights_conv1)


# ## Perform 10,000 optimization iterations
# 
# We now perform 10,000 optimization iterations and abort the optimization if no improvement is found on the validation-set in 1000 iterations.
# 
# An asterisk * is shown if the classification accuracy on the validation-set is an improvement.

# In[44]:

optimize(num_iterations=10000)


# In[45]:

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


# The convolutional weights have now been optimized. Compare these to the random weights shown above. They appear to be almost identical. In fact, I first thought there was a bug in the program because the weights look identical before and after optimization.
# 
# But try and save the images and compare them side-by-side (you can just right-click the image to save it). You will notice very small differences before and after optimization.
# 
# The mean and standard deviation has also changed slightly, so the optimized weights must be different.

# In[46]:

plot_conv_weights(weights=weights_conv1)


# ## Initialize Variables Again
# 
# Re-initialize all the variables of the neural network with random values.

# In[47]:

init_variables()


# This means the neural network classifies the images completely randomly again, so the classification accuracy is very poor because it is like random guesses.

# In[48]:

print_test_accuracy()


# The convolutional weights should now be different from the weights shown above.

# In[49]:

plot_conv_weights(weights=weights_conv1)


# ## Restore Best Variables
# 
# Re-load all the variables that were saved to file during optimization.

# In[50]:

saver.restore(sess=session, save_path=save_path)


# The classification accuracy is high again when using the variables that were previously saved.
# 
# Note that the classification accuracy may be slightly higher or lower than that reported above, because the variables in the file were chosen to maximize the classification accuracy on the validation-set, but the optimization actually continued for another 1000 iterations after saving those variables, so we are reporting the results for two slightly different sets of variables. Sometimes this leads to slightly better or worse performance on the test-set.

# In[51]:

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


# The convolutional weights should be nearly identical to those shown above, although not completely identical because the weights shown above had 1000 optimization iterations more.

# In[52]:

plot_conv_weights(weights=weights_conv1)


# ## Close TensorFlow Session

# We are now done using TensorFlow, so we close the session to release its resources.

# In[53]:

# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
# session.close()


# ## Conclusion
# 
# This tutorial showed how to save and retrieve the variables of a neural network in TensorFlow. This can be used in different ways. For example, if you want to use a neural network for recognizing images then you only have to train the network once and you can then deploy the finished network on other computers.
# 
# Another use of checkpoints is if you have a very large neural network and data-set, then you may want to save checkpoints at regular intervals in case the computer crashes, so you can continue the optimization at a recent checkpoint instead of having to restart the optimization from the beginning.
# 
# This tutorial also showed how to use the validation-set for so-called Early Stopping, where the optimization was aborted if it did not regularly improve the validation error. This is useful if the neural network starts to overfit and learn the noise of the training-set; although it was not really an issue with the convolutional network and MNIST data-set used in this tutorial.
# 
# An interesting observation was that the convolutional weights (or filters) changed very little from the optimization, even though the performance of the network went from random guesses to near-perfect classification. It seems strange that the random weights were almost good enough. Why do you think this happens?

# ## Exercises
# 
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
# 
# You may want to backup this Notebook before making any changes.
# 
# * Optimization is stopped after 1000 iterations without improvement. Is this enough? Can you think of a better way to do Early Stopping? Try and implement it.
# * If the checkpoint file already exists then load it instead of doing the optimization.
# * Save a new checkpoint for every 100 optimization iterations. Retrieve the latest using `saver.latest_checkpoint()`. Why would you want to save multiple checkpionts instead of just the most recent?
# * Try and change the neural network, e.g. by adding another layer. What happens when you reload the variables from a different network?
# * Plot the weights for the 2nd convolutional layer before and after optimization using the function `plot_conv_weights()`. Are they almost identical as well?
# * Why do you think the optimized convolutional weights are almost the same as the random initialization?
# * Remake the program yourself without looking too much at this source-code.
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
