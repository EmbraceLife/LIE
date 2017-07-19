"""
floyd_guide

## how to start from my working environment
wktf

# how to login and logout floyd
floyd login
floyd logout

# how to upload dataset from local to floyd
cd /Users/Natsume/Downloads/data_for_all/stocks/features_targets # where data files stored in
floyd data init indices_train # as data dirname in floyd data directory
floyd data upload

# how to use dataset from floyd, inside training.py
features = bz_load_array("/input/train_features_path") # not features_targets_train_val_test/train_features_path

# get dataset ID
floyd data status

# delete dataset
floyd data delete ID

# how to update the files to be run on floyd
cd dir_to_run_on_floyd
floyd init new_name_dir_for_floyd
floyd run --data dataID --env keras --gpu "python file_to_run.py"


# test before training
floyd run --env keras --data oWCUSE9DFBy4X8TY6kw4Tg --gpu "python prep_data_03_stock_04_load_train_valid_test_features_targets_arrays_from_files_for_train.py"

# train on floyd
floyd run --data oWCUSE9DFBy4X8TY6kw4Tg --gpu "python build_model_03_stock_03_train_evaluate_save_best_model_in_training.py " --env keras

# check how far and how well the model is training on floyd
floyd logs daniel/train_revised_version/8

# how to stop a running file on floyd
- go to Floyd/Experiments
- find the latest experiment to stop or check details

# how to find output and input on Floyd
- go to Floyd/Data
- find the latest input or output
"""


#### Test the floyd trained model and log

from tensorflow.contrib.keras.python.keras.models import load_model

from build_model_03_stock_02_build_compile_model_with_WindPuller_init import wp

floyd_model = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/model_2epoch.h5")

floyd_model.model.summary()

floyd_model.model.get_weights()

# tensorboard --logdir "/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/log"
