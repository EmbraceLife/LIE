################################
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
# to write multiple lines inside pdb
# !import code; code.interact(local=vars())

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math


# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.*********
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10


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


# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


def new_weights(shape): # set weights with sd of 0.05, make all closer to 0 but not 0
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length): # set bias to be 0.05
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters. (as this layer's output number?)
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters] # 4 dimension tensor
    # first 2-d == this layer's filter image's width-height
    # 3rd dimension == previous layer's output images number
    # 4th dimension == this layer's output images number


    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)
    # 4-d shape is 5x5 (filter-image 5x5)  x1 (num-input-channel: gray == 1)  x16  (we want to output 16 channels)


    # Create new biases, one for each filter, each filter for each output image
    biases = new_biases(length=num_filters)


    layer = tf.nn.conv2d(input=input, # input is the previous layer's output
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME') ##################################### attention: what does padding look like

    layer += biases

    if use_pooling:

        layer = tf.nn.max_pool(value=layer, # current convol layer
                               ksize=[1, 2, 2, 1], # shape of pooling screen window? ###################### attention
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights



def flatten_layer(layer): # layer is the input layer to this to-be flatten_layer

    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer



x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


x_image = tf.reshape(x, [-1, img_size, img_size, num_channels]) # still num_color == 1?  ################## attention


y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')


y_true_cls = tf.argmax(y_true, dimension=1)


layer_conv1, weights_conv1 =     new_conv_layer(input=x_image, #1, shape set as [-1, img_size, img_size, num_channels]
                   num_input_channels=num_channels, #2, set as 1 at beginning
                   filter_size=filter_size1,#3, set as 5 at beginning
                   num_filters=num_filters1, #4, set as 16 at beginning
                   use_pooling=True) #5




layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1, # previous convol layer as input
                   num_input_channels=num_filters1, # num of output images of previous layer as num of input channels
                   filter_size=filter_size2, # set as 5 at beginning
                   num_filters=num_filters2, # set as 36 at beginning
                   use_pooling=True)



layer_flat, num_features = flatten_layer(layer_conv2)



layer_fc1 = new_fc_layer(input=layer_flat, # from 2nd-convol layer being flattened
                         num_inputs=num_features, # all convol images/channels flattened
                         num_outputs=fc_size, # set as 128 at beginning
                         use_relu=True)



layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size, # set as 128
                         num_outputs=num_classes, # set as 10 at beginning
                         use_relu=False) # not use relu for the output layer



y_pred = tf.nn.softmax(layer_fc2) # use softmax for the output layer


y_pred_cls = tf.argmax(y_pred, dimension=1) # convert from one-hot encoding form to normal form



# this function take layer_fc2 as input directly
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)


cost = tf.reduce_mean(cross_entropy)


optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


correct_prediction = tf.equal(y_pred_cls, y_true_cls)


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session = tf.Session()


session.run(tf.global_variables_initializer())


train_batch_size = 64



total_iterations = 0

def optimize(num_iterations):

    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()


    for i in range(total_iterations,  # maybe we could try progress bar here
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size) # tensorflow take care of randomly batch#####

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the current batch dataset.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

#
# def plot_example_errors(cls_pred, correct):
#     # This function is called from print_test_accuracy() below.
#
#     # cls_pred is an array of the predicted class-number for
#     # all images in the test-set.
#
#     # correct is a boolean array whether the predicted class
#     # is equal to the true class for each image in the test-set.
#
#     # Negate the boolean array.
#     incorrect = (correct == False)
#
#     # Get the images from the test-set that have been
#     # incorrectly classified.
#     images = data.test.images[incorrect]
#
#     # Get the predicted classes for those images.
#     cls_pred = cls_pred[incorrect]
#
#     # Get the true classes for those images.
#     cls_true = data.test.cls[incorrect]
#
#     # Plot the first 9 images.
#     plot_images(images=images[0:9],
#                 cls_true=cls_true[0:9],
#                 cls_pred=cls_pred[0:9])
#
#
#
# def plot_confusion_matrix(cls_pred):
#     # This is called from print_test_accuracy() below.
#
#     # cls_pred is an array of the predicted class-number for
#     # all images in the test-set.
#
#     # Get the true classifications for the test-set.
#     cls_true = data.test.cls
#
#     # Get the confusion matrix using sklearn.
#     cm = confusion_matrix(y_true=cls_true,
#                           y_pred=cls_pred)
#
#     # Print the confusion matrix as text.
#     print(cm)
#
#     # Plot the confusion matrix as an image.
#     plt.matshow(cm)
#
#     # Make various adjustments to the plot.
#     plt.colorbar()
#     tick_marks = np.arange(num_classes)
#     plt.xticks(tick_marks, range(num_classes))
#     plt.yticks(tick_marks, range(num_classes))
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#
#     # Ensure the plot is shown correctly with multiple plots
#     # in a single Notebook cell.
#     plt.show()


test_batch_size = 256
#
# def print_test_accuracy(show_example_errors=False,
#                         show_confusion_matrix=False):
#
#     # Number of images in the test-set.
#     num_test = len(data.test.images)
#
#     # Allocate an array for the predicted classes which
#     # will be calculated in batches and filled into this array.
#     cls_pred = np.zeros(shape=num_test, dtype=np.int)
#
#     # Now calculate the predicted classes for the batches.
#     # We will just iterate through all the batches.
#     # There might be a more clever and Pythonic way of doing this.
#
#     # The starting index for the next batch is denoted i.
#     i = 0
#
#     # loop through the entire test set using mini-batches, to make predictions by batches
#     while i < num_test:
#         # The ending index for the next batch is denoted j.
#         j = min(i + test_batch_size, num_test)
#
#         # Get the images from the test-set between index i and j.
#         images = data.test.images[i:j, :]
#
#         # Get the associated labels.
#         labels = data.test.labels[i:j, :]
#
#         # Create a feed-dict with these images and labels.
#         feed_dict = {x: images,
#                      y_true: labels}
#
#         # Calculate the predicted class using TensorFlow.
#         cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
#
#         # Set the start-index for the next batch to the
#         # end-index of the current batch.
#         i = j
#
#     # Convenience variable for the true class-numbers of the test-set.
#     cls_true = data.test.cls
#
#     # Create a boolean array whether each image is correctly classified.
#     correct = (cls_true == cls_pred)
#
#     # Calculate the number of correctly classified images.
#     # When summing a boolean array, False means 0 and True means 1.
#     correct_sum = correct.sum()
#
#     # Classification accuracy is the number of correctly classified
#     # images divided by the total number of images in the test-set.
#     acc = float(correct_sum) / num_test
#
#     # Print the accuracy.
#     msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
#     print(msg.format(acc, correct_sum, num_test))
#
#     # Plot some examples of mis-classifications, if desired.
#     if show_example_errors:
#         print("Example errors:")
#         plot_example_errors(cls_pred=cls_pred, correct=correct)
#
#     # Plot the confusion matrix, if desired.
#     if show_confusion_matrix:
#         print("Confusion Matrix:")
#         plot_confusion_matrix(cls_pred=cls_pred)
#

# print_test_accuracy()



optimize(num_iterations=1)

# print_test_accuracy()

#
# optimize(num_iterations=99) # We already performed 1 iteration above.
#
# print_test_accuracy(show_example_errors=True)
#
#
# optimize(num_iterations=900) # We performed 100 iterations above.
#
#
# print_test_accuracy(show_example_errors=True)
#
#
# optimize(num_iterations=9000) # We performed 1000 iterations above.
#
# print_test_accuracy(show_example_errors=True,
#                     show_confusion_matrix=True)
#
#

def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights) # just Retrieve value, not calculating here

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot: height = weight = num of grids
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    # loop through every subplot using ax
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i] # [:, :, ....] is width, height of image
            # weights has 4-d shape, e.g. 5x5 (filter-image 5x5)  x1 (num-input-channel: gray == 1)  x16  (we want to output 16 channels)

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    #  **Calculate and retrieve** the output values of the layer using the current weights and biases
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)
    # values == current convol layer
    # convol_layer_1  is <tf.Tensor 'Relu:0' shape=(?, 14, 14, 16) dtype=float32>

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]
            # 0 cos there is only one image
            # i refers to index of output channels/images of this convol layer

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

#
#
# def plot_image(image):
#     plt.imshow(image.reshape(img_shape),
#                interpolation='nearest',
#                cmap='binary')
#
#     plt.show()
#

image1 = data.test.images[0]



image2 = data.test.images[13]



# plot_conv_weights(weights=weights_conv1)

set_trace()
plot_conv_layer(layer=layer_conv1, image=image1)


# plot_conv_layer(layer=layer_conv1, image=image2)



# plot_conv_weights(weights=weights_conv2, input_channel=0)


# plot_conv_weights(weights=weights_conv2, input_channel=1)



plot_conv_layer(layer=layer_conv2, image=image1)


# And these are the results of applying the filter-weights to the second image.

# In[66]:

# plot_conv_layer(layer=layer_conv2, image=image2)
