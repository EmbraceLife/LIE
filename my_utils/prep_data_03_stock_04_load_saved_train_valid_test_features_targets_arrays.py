"""
Inputs:
1. file_path for feature array, target array for training, validation, test set

Return:
1. ready-to-train dataset: features arrays, targets arrays for training, validation, test set


### Summary:
- to load is faster than to process and create features and targets from previous pyfile
- processed ready features array and targets array for training, validation, test sets can be imported from this source file

### Steps
- load train_features, train_targets, valid_features, valid_targets, test_features, test_targets from files
- change its shape for training and predict
"""


from prep_data_98_funcs_save_load_large_arrays import bz_load_array
import numpy as np
# create paths for loading those arrays above
train_features_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/train_features_path"
train_targets_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/train_targets_path"

valid_features_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/valid_features_path"
valid_targets_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/valid_targets_path"

test_features_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/test_features_path"
test_targets_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/test_targets_path"

train_features = bz_load_array(train_features_path)
train_targets = bz_load_array(train_targets_path)
valid_features = bz_load_array(valid_features_path)
valid_targets = bz_load_array(valid_targets_path)
test_features = bz_load_array(test_features_path)
test_targets = bz_load_array(test_targets_path)

train_features = np.reshape(train_features, (-1, 30, 61))
valid_features = np.reshape(valid_features, (-1, 30, 61))
test_features = np.reshape(test_features, (-1, 30, 61))
