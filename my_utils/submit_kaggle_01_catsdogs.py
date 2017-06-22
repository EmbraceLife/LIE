from prep_data_98_funcs_save_load_large_arrays import bz_load_array
trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"

#####################################
# clip predictions
# load test_predictions on training set total 200 samples
preds_test = bz_load_array(trained_model_path+"/preds_test")
# If second column is 1, it's a dog, otherwise cat
isdog = preds_test[:,1]
print("Raw Predictions: " + str(isdog[:5]))
print("Mid Predictions: {}".format(isdog[(isdog < .6) & (isdog > .4)]))
# %d to display a number, {} can display anything like array or number
print("Edge Predictions: {}".format((isdog[(isdog == 1) | (isdog == 0)])))

###############################################
# clip every value above 0.95 to 0.95, and every value below 0.05 to 0.05
isdog = isdog.clip(min=0.05, max=0.95) # not 0.95, display as 0.9499999
isdog = isdog.round(2) # not help at all


###############################################
# organise submit dataset or array
from prep_data_02_img_folder_2_iterators import test_batches
import numpy as np
# Extract imageIds from the filenames in our test/unknown directory
filenames = test_batches.filenames
names=[]
for f in filenames:
	names.append(int(f[8:f.find('.')]))

# cbind ids and preds together into a single array
ids = np.array(names)
subm = np.stack([ids,isdog], axis=1)
