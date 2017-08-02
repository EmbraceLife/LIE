"""
load_train_valid_test_features_target_arrays
- load these arrays from files before training

Uses:
1. load train, valid, test pairs for feature array and target array from files
2. transpose its dimensions from (0, 1, 2) to (0, 2, 1)

Inputs:
1. file_path for feature array, target array for training, validation, test set

Return:
1. features arrays, targets arrays for training, validation, test set


"""


from save_load_large_arrays import bz_load_array
import numpy as np
# create paths for loading those arrays above

########################################################################
#### 1. load data from local folders, training locally
########################################################################
# train_features_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/train_features_path"
# train_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/train_targets_path"
#
# valid_features_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/valid_features_path"
# valid_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/valid_targets_path"

############### No need for test set for now ##################
### test_features_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/test_features_path"
### test_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/test_targets_path"


########################################################################
##### 1. loading data from Floyd input folder, For floyd training
########################################################################

train_features_path = "/input/train_features_path"
train_targets_path = "/input/train_targets_path"

valid_features_path = "/input/valid_features_path"
valid_targets_path = "/input/valid_targets_path"

############### No need for test set for now ##################
#### test_features_path = "/input/test_features_path"
#### test_targets_path = "/input/test_targets_path"




########################################################################
##### 2. loading data into training environment
########################################################################

train_features = bz_load_array(train_features_path)
train_targets = bz_load_array(train_targets_path)
valid_features = bz_load_array(valid_features_path)
valid_targets = bz_load_array(valid_targets_path)

############### No need for test set for now ##################
# test_features = bz_load_array(test_features_path)
# test_targets = bz_load_array(test_targets_path)

##################################################################
### do np.reshape and np.transpose have very different result
## train_features_reshape = np.reshape(train_features, (-1, 30, 61))
## valid_features_reshape = np.reshape(valid_features, (-1, 30, 61))
## test_features_reshape = np.reshape(test_features, (-1, 30, 61))
##################################################################
train_features = np.transpose(train_features, (0, 2, 1))
valid_features = np.transpose(valid_features, (0, 2, 1))

############### No need for test set for now ##################
# test_features = np.transpose(test_features, (0, 2, 1))

print("train_features, train_targets, shapes: \n")
print(train_features.shape)
print(train_targets.shape)
print(valid_features.shape)
print(valid_targets.shape)
print("-----------------------------------")
print("first row of features:")
print(train_features[0])
print("first 100 values of train_targets:")
print(train_targets[0])
print("-----------------------------------")


###### Test floyd on this file
# floyd run --env keras --data DJeKLuEpYqJPBYhxViyRfm --gpu "python prep_data_03_stock_04_load_train_valid_test_features_targets_arrays_from_files_for_train.py"
