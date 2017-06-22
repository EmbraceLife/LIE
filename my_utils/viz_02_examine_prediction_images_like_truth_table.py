from numpy.random import permutation
import numpy as np
# import plotting funcs
from viz_01_display_n_images_together import plots_idx

# load sample prediction array using bcolz
from prep_data_98_funcs_save_load_large_arrays import bz_load_array
trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
img_dir = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/train/"
# load train_batches
from prep_data_02_img_folder_2_iterators import train_batches
# load test_predictions on training set total 200 samples
preds_train = bz_load_array(trained_model_path+"/preds_train")

# access true labels of training set
true_labels = train_batches.classes
# access filenames of training set samples
filenames = train_batches.filenames
# which cats and dogs number representation
train_batches.class_indices
# {'cats': 0, 'dogs': 1}: cat label as 0, dog label as 1
# (0, 1) as dog, 1 as high, 0 as low: high prob as dog, low prob as cat ?
# (1, 0) as cat, 1 as high, 0 as low: high prob as cat, low prob as dog ?

# access predictions of training set samples
our_cat_predictions = preds_train[:,0]
# if our_cat_predictions = 1, it is cat; if our_cat_predictions = 0, it is dog
our_dog_predictions = preds_train[:,1]
# if our_dog_predictions = 0, it is cat; if our_dog_predictions = 1, it is dog
our_cat_labels = np.round(1-our_cat_predictions)
# inside our_cat_labels: 0 refers to cats class, 1 refers to dogs class
our_dog_labels = np.round(our_dog_predictions)

# view 4 images at once
num_images = 4

#1. A few correct labels at random
correct = np.where(our_cat_labels==true_labels.astype('float32'))[0]
print("Found %d correct labels" % len(correct))
idx = permutation(correct)[:num_images]
# title is cat probability: close to 1, is higher prob to cat, close to 0 is lower prob to cat
plots_idx(img_dir, filenames, idx, titles=our_cat_predictions[idx])


#2. A few incorrect labels at random
incorrect = np.where(our_cat_labels!=true_labels.astype('float32'))[0]
print("Found %d incorrect labels" % len(incorrect))
idx = permutation(incorrect)[:num_images]
# title: cat probability: close to 1, high prob to cat; close to 0, lower prob to cat
plots_idx(img_dir, filenames, idx, titles=our_cat_predictions[idx])


#3a. The images we most confident were cats, and are actually cats
correct_cats = np.where((our_cat_labels==0.) & (our_cat_labels==true_labels.astype('float32')))[0]
print("Found %d confident correct cats labels" % len(correct_cats))

most_correct_cats = np.argsort(our_cat_predictions[correct_cats])[::-1][:num_images]
# title: cat probability: close to 1, high prob to cat; close to 0, lower prob to cat
plots_idx(img_dir, filenames, idx=correct_cats[most_correct_cats], titles=our_cat_predictions[correct_cats][most_correct_cats])

#3b. The images we most confident were dogs, and are actually dogs
correct_dogs = np.where((our_cat_labels==1.) & (our_cat_labels==true_labels.astype('float32')))[0]
print("Found %d confident correct dogs labels" % len(correct_dogs))
most_correct_dogs = np.argsort(our_cat_predictions[correct_dogs])[:num_images]
# title: cat probability: close to 1, high prob to cat; close to 0, lower prob to cat
plots_idx(img_dir, filenames, correct_dogs[most_correct_dogs], our_cat_predictions[correct_dogs][most_correct_dogs])

correct_dogs = np.where((our_dog_labels==1.) & (our_dog_labels==true_labels.astype('float32')))[0]
print("Found %d confident correct dogs labels" % len(correct_dogs))
most_correct_dogs = np.argsort(our_dog_predictions[correct_dogs])[:num_images]
# title: dog probability: close to 1, high prob to dog; close to 0, lower prob to dog
plots_idx(img_dir, filenames, correct_dogs[most_correct_dogs], our_dog_predictions[correct_dogs][most_correct_dogs])

#4a. The images we were most confident were cats, but are actually dogs
incorrect_cats = np.where((our_cat_labels==0.0) & (our_cat_labels!=true_labels.astype('float32')))[0]
print("Found %d incorrect cats" % len(incorrect_cats))
if len(incorrect_cats):
    most_incorrect_cats = np.argsort(our_cat_predictions[incorrect_cats])[::-1][:num_images]
    plots_idx(img_dir, filenames, incorrect_cats[most_incorrect_cats], our_cat_predictions[incorrect_cats][most_incorrect_cats])


#4b. The images we were most confident were dogs, but are actually cats
incorrect_dogs = np.where((our_dog_labels==1.0) & (our_dog_labels != true_labels.astype('float32')))[0]
print("Found %d incorrect dogs" % len(incorrect_dogs))
if len(incorrect_dogs):
    most_incorrect_dogs = np.argsort(our_dog_predictions[incorrect_dogs])[:num_images]
    plots_idx(img_dir, filenames, incorrect_dogs[most_incorrect_dogs], our_dog_predictions[incorrect_dogs][most_incorrect_dogs])
