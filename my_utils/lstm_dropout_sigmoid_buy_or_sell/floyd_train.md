"""
floyd_guide

## how to start from my working environment
`wktf`


# how to login and logout floyd
```
floyd login
floyd logout
```

# prepare training and validation dataset for training
- decide how much dataset used for training
- how much for validation and testing
	- go inside `combination_multiple_linear_functions.py`
	- change values for `days_for_valid` and `days_for_test`
	- get a new folder for this new datasets
- run the following code to prepare training, validation and test sets
```
wktf
cd lstm_dropout_sigmoid_1
#
python -m pdb combination_multiple_linear_functions.py
```

# how to upload dataset from local to floyd
- decide which dataset use to train
- go to the folder for training dataset
- init and upload dataset with the name above
```
cd /Users/Natsume/Downloads/data_for_all/stocks/features_targets
cd /Users/Natsume/Downloads/data_for_all/stocks/features_targets1
floyd data init lstm_sigmoid_1_dataset
floyd data upload
```
# how to use dataset from floyd, inside training.py
- use the data name defined above
- do not use the dataset names used on your local computer
```python
features = bz_load_array("/input/to_give_a_training_data_name")
```

# get dataset ID
`floyd data status`

# delete dataset
`floyd data delete ID`

# how to update the files to be run on floyd
```
cd dir_to_run_on_floyd
floyd init lstm_sigmoid_1_files

```

# get training datasets ready before floyd train
- prepare `floyd_requirement.txt` for import necessary libraries onto floyd
- go to file `get_train_valid_datasets.py`
	- switching between datasets folders locally and dataset folders on floyd
```
floyd run --env keras --data JnGrjujE96R2C4eiUG5Xz5 --gpu "python get_train_valid_datasets.py"
```

# get ready to train on floyd
- go to file `model_train.py`
- change model folder by switching between local folder and floyd folder
- change log folder by switching between local folder and floyd folder
```
floyd run --data L2d6JLFS2TpxqxkqJNA6xm --gpu "python train_model.py " --env keras
```

# check how far and how well the model is training on floyd
`floyd logs daniel/train_revised_version/8`

# how to stop a running file on floyd
- go to Floyd/Experiments
- find the latest experiment to stop or check details

# how to find output and input on Floyd
- go to Floyd/Data
- find the latest input or output



#### Test the floyd trained model and log
```python
from tensorflow.contrib.keras.python.keras.models import load_model

from build_model_03_stock_02_build_compile_model_with_WindPuller_init import wp

floyd_model = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/model_2epoch.h5")

floyd_model.model.summary()
floyd_model.model.get_weights()
```

#### checkout the log
```
tensorboard --logdir "/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/log"
```
