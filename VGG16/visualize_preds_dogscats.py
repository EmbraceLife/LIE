
## access preds_val
from save_load_large_array import bz_load_array
preds_val = bz_load_array('/Users/Natsume/Downloads/data_for_all/dogscats/results/preds_val')

## access test_batches
from vgg16_iterator_from_directory import val_batches

# get predicted labels (1 as cat, 0 as not cat)
import numpy as np
our_predictions = preds_val[:,0]
our_labels = np.round(1-our_predictions)

# access image labels and filenames with order as they are in folders
true_labels = val_batches.classes
filenames = val_batches.filenames

# Number of images to plot
n_view = 4

#1. A few correct labels at random
correct = np.where(our_labels==true_labels)[0]
print("Found %d correct labels" % len(correct))
idx = permutation(correct)[:n_view]
plots_idx(idx, our_predictions[idx])

# #2. A few incorrect labels at random
# incorrect = np.where(our_labels!=true_labels)[0]
# print "Found %d incorrect labels" % len(incorrect)
# idx = permutation(incorrect)[:n_view]
# plots_idx(idx, our_predictions[idx])
#
# #3a. The images we most confident were cats, and are actually cats
# correct_cats = np.where((our_labels==0) & (our_labels==true_labels))[0]
# print "Found %d confident correct cats labels" % len(correct_cats)
# most_correct_cats = np.argsort(our_predictions[correct_cats])[::-1][:n_view]
# plots_idx(correct_cats[most_correct_cats], our_predictions[correct_cats][most_correct_cats])
#
# #3b. The images we most confident were dogs, and are actually dogs
# correct_dogs = np.where((our_labels==1) & (our_labels==true_labels))[0]
# print "Found %d confident correct dogs labels" % len(correct_dogs)
# most_correct_dogs = np.argsort(our_predictions[correct_dogs])[:n_view]
# plots_idx(correct_dogs[most_correct_dogs], our_predictions[correct_dogs][most_correct_dogs])
#
# #4a. The images we were most confident were cats, but are actually dogs
# incorrect_cats = np.where((our_labels==0) & (our_labels!=true_labels))[0]
# print "Found %d incorrect cats" % len(incorrect_cats)
# if len(incorrect_cats):
#     most_incorrect_cats = np.argsort(our_predictions[incorrect_cats])[::-1][:n_view]
#     plots_idx(incorrect_cats[most_incorrect_cats], our_predictions[incorrect_cats][most_incorrect_cats])
#
# #4b. The images we were most confident were dogs, but are actually cats
# incorrect_dogs = np.where((our_labels==1) & (our_labels!=true_labels))[0]
# print "Found %d incorrect dogs" % len(incorrect_dogs)
# if len(incorrect_dogs):
#     most_incorrect_dogs = np.argsort(our_predictions[incorrect_dogs])[:n_view]
#     plots_idx(incorrect_dogs[most_incorrect_dogs], our_predictions[incorrect_dogs][most_incorrect_dogs])
#
#
# #5. The most uncertain labels (ie those with probability closest to 0.5).
# most_uncertain = np.argsort(np.abs(our_predictions-0.5))
# plots_idx(most_uncertain[:n_view], our_predictions[most_uncertain])
#
#
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(true_labels, our_labels)
#
# plot_confusion_matrix(cm, val_batches.class_indices)
