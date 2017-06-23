from tensorflow.contrib.keras.python.keras.models import load_model
from build_model_03_stock_02_build_compile_model_with_wp_init import wp
from prep_data_03_stock_04_load_saved_train_valid_test_features_targets_arrays import train_features, train_targets, valid_features, valid_targets, test_features, test_targets
import numpy as np


# load the best model so far, using the saved best model by author, not the one trained above
wp = wp.load_model("/Users/Natsume/Downloads/DeepTrade_keras/author_log_weights/model.30.best")

# predict with model on training, validation and test sets
test_preds_pos = wp.predict(test_features, 1024)
test_targets = np.reshape(test_targets, (-1, 1))
test_preds_pos = np.reshape(test_preds_pos, (-1, 1))
test_pos_targets = np.concatenate((test_preds_pos, test_targets), axis=1)

valid_preds_pos = wp.predict(valid_features, 1024)
valid_targets = np.reshape(valid_targets, (-1, 1))
valid_preds_pos = np.reshape(valid_preds_pos, (-1, 1))
valid_pos_targets = np.concatenate((valid_preds_pos, valid_targets), axis=1)

train_preds_pos = wp.predict(train_features, 1024)
train_targets = np.reshape(train_targets, (-1, 1))
train_preds_pos = np.reshape(train_preds_pos, (-1, 1))
train_pos_targets = np.concatenate((train_preds_pos, train_targets), axis=1)

# save predictions_pos and price_change_pct on training, validation, test sets
from prep_data_98_funcs_save_load_large_arrays import bz_save_array
train_pos_targets_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/train_pos_targets"
valid_pos_targets_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/valid_pos_targets"
test_pos_targets_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/test_pos_targets"

bz_save_array(train_pos_targets_path, train_pos_targets)
bz_save_array(valid_pos_targets_path, valid_pos_targets)
bz_save_array(test_pos_targets_path, test_pos_targets)
