# source from https://github.com/happynoom/DeepTrade_keras
"""
extract_features_target_from_csv_multiple
- from a stock csv file, extract features array, target array
- split each array into train, valid, test 3 sub_arrays groups
- do the same two steps to another csv
- rbind 3 new sub_arrays onto previous 3 sub_arrays groups
- return 3 large arrays groups (each group has features array, target array)
- total 6 arrays are saved in 6 files
- this file work specifically on indices dataset



Uses:
1. run this file, to turn a number of csv files, into train_features_array, train_target_array, valid_features_array, valid_target array, test_features_array, test_target_array
2. save thest 6 arrays into folders with bz_save_array


Inputs:
- a dir_path with a number of csv files
- a user selected indicators supported internally
- 6 dir_paths to save the 6 arrays above

Returns:
- 6 arrays above
- 6 folders with the 6 arrays inside


### Steps
- import read_csv_2_arrays to convert csv to arrays
- import extract_feature to convert arrays of OHLCV to features, targets arrays
- import bz_save_array to save large arrays
- set days for training, validation and testing
- create gobal variables for storing training, validation and testing features arrays and targets arrays
- create paths for storing those arrays above
- user selected indicators for converting OHLCV to
- dir_path for stock CSVs
- count number of csv to use for creating features array and target arrays
- loop through every csv to convert from csv to arrays OHLCV, to arrays features and targets, and concatenate features and targets of different csv files
"""

import os
from stock_csv_object_array import read_csv_2_arrays
from stock_csv_pandas_array import csv_df_arrays
from ohlcv_to_features_targets import extract_feature
import numpy as np
from save_load_large_arrays import bz_save_array

# set days for training, validation and testing
days_for_valid = 700 # 1000 is too much dataset to be taken away from training
days_for_test = 0 # number of test samples
# input_shape = [30, 24] # [30, 61]  # [length of time series, length of feature]
# window = input_shape[0]

# create gobal variables for storing training, validation and testing features arrays and targets arrays
train_features = None
valid_targets = None
valid_features = None
train_targets = None
test_features = None
test_targets = None

# create paths for storing those arrays above
train_features_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/train_features_path"
valid_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/valid_targets_path"
valid_features_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/valid_features_path"
train_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/train_targets_path"
test_features_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/test_features_path"
test_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets/test_targets_path"

########################################################################
### How many indicators do we use to create features
########################################################################
### Situation 1: use 61 indicators without using OHLC as features
# user selected indicators for converting OHLCV to
user_indicators = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"]

### Situation 2: 20 MA indicators + 4 OHLC as indicators
# user_indicators = ["MA"] # only select MA,
# and OHLC as features are added by comment free a 4 lines of code in  'prep_data_03_stock_02_OHLCV_arrays_2_features_targets_arrays.py'

# dir_path for stock csv
dataset_dir = "/Users/Natsume/Downloads/data_for_all/stocks/dataset"

# count number of csv to use for creating features array and target arrays
total_csv_combine = 25
current_num_csv = 0

# loop through every csv to convert from csv to arrays OHLCV, to arrays features and targets, and concatenate features and targets of different csv files
for filename in os.listdir(dataset_dir):
    if current_num_csv >= total_csv_combine:
	    break
	# 000001.csv must be the first file accessed by program
    if filename == '000001.csv':

	    print("processing the first file: " + filename)
	    filepath = dataset_dir + "/" + filename
	    # _, _, opens, highs, lows, closes, volumes = read_csv_2_arrays(filepath)

		# opens, highs, lows, closes, volumes are to be used inside extract_feature to create indicators_features, price_change_targets
	    moving_features, moving_targets = extract_feature(selector=user_indicators, file_path=filepath)


		# save test_set and train_set
		# valid_set: 1000 days
		# test_set: 700 days
		# train_set: 6434 - 1000 -700 days
	    print("split feature array and target arrays to train, valid, test sets...")
	    train_end_test_begin = moving_features.shape[0] - days_for_valid - days_for_test

	    train_features = moving_features[0:train_end_test_begin]
	    train_targets = moving_targets[0:train_end_test_begin]

	    valid_features = moving_features[train_end_test_begin:train_end_test_begin+days_for_valid]
	    valid_targets = moving_targets[train_end_test_begin:train_end_test_begin+days_for_valid]

	    test_features = moving_features[train_end_test_begin+days_for_valid:train_end_test_begin+days_for_valid+days_for_test]
	    test_targets = moving_targets[train_end_test_begin+days_for_valid:train_end_test_begin+days_for_valid+days_for_test]

    else:
	    print("processing file (start counting from 0) no. %d: " % current_num_csv + filename)
	    filepath = dataset_dir + "/" + filename
	    # _, _, opens, highs, lows, closes, volumes = read_csv_2_arrays(filepath)

	    moving_features, moving_targets = extract_feature(selector=user_indicators, file_path=filepath)

	    print("split feature array and target arrays to train, valid, test sets...")
	    train_end_test_begin = moving_features.shape[0] - days_for_valid - days_for_test

	    train_features_another = moving_features[0:train_end_test_begin]
	    train_targets_another = moving_targets[0:train_end_test_begin]

	    valid_features_another = moving_features[train_end_test_begin:train_end_test_begin+days_for_valid]
	    valid_targets_another = moving_targets[train_end_test_begin:train_end_test_begin+days_for_valid]

	    test_features_another = moving_features[train_end_test_begin+days_for_valid:train_end_test_begin+days_for_valid+days_for_test]
	    test_targets_another = moving_targets[train_end_test_begin+days_for_valid:train_end_test_begin+days_for_valid+days_for_test]

		# rbind different csv's train, val and test together
	    train_features = np.concatenate((train_features, train_features_another), axis = 0)
	    train_targets = np.concatenate((train_targets, train_targets_another), axis = 0)

	    valid_features = np.concatenate((valid_features, valid_features_another), axis = 0)
	    valid_targets = np.concatenate((valid_targets, valid_targets_another), axis = 0)

	    test_features = np.concatenate((test_features, test_features_another), axis = 0)
	    test_targets = np.concatenate((test_targets, test_targets_another), axis = 0)

    current_num_csv += 1

print("training: (%d, %d, %d)" % (train_features.shape[0], train_features.shape[1], train_features.shape[2]))
print("validation: (%d, %d, %d)" % (valid_features.shape[0], valid_features.shape[1], valid_features.shape[2]))
print("test: (%d, %d, %d)" % (test_features.shape[0], test_features.shape[1], test_features.shape[2]))

bz_save_array(train_features_path, train_features)
bz_save_array(train_targets_path, train_targets)
bz_save_array(valid_features_path, valid_features)
bz_save_array(valid_targets_path, valid_targets)
bz_save_array(test_features_path, test_features)
bz_save_array(test_targets_path, test_targets)
