# source from https://github.com/happynoom/DeepTrade_keras
"""
### workflow

1. use a number of stock data like 000001.csv, and others with same date range
1. put these csv files in the same dir "/Users/Natsume/Downloads/DeepTrade_keras/dataset"
2. convert csv to a list of objects containing date, OHLCV
3. convert list of objects to feature array and target array
	1. 6464 days OHLCV, converted to array of 61 features, shape (61, 6464);
	2. split (61, 6464) into a list of 6434 of (61, 30);
	3. then convert it to array shape (6434, 61*30);
	4. but why not (6434, 61, 30)
3. stack each stock.csv feature and target array on one another to make a large feature array and target array
4. split feature array and target array into train and test set
5. save them in separate files (train and test)
"""

import os
from prep_data_03_stock_csv_2_list_objects import RawData, read_sample_data
# from dataset import DataSet # not sure it is actually used here
from prep_data_03_funcs_class_4_OHLCV_2_indicators import extract_feature
import numpy
if __name__ == '__main__':
    days_for_test = 700 # number of test samples
    input_shape = [30, 61]  # [length of time series, length of feature]
    window = input_shape[0]
    fp = open("ultimate_feature.%s" % window, "w")
    lp = open("ultimate_label.%s" % window, "w")
    fpt = open("ultimate_feature.test.%s" % window, "w")
    lpt = open("ultimate_label.test.%s" % window, "w")

    selector = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"] # indicators groups?
    dataset_dir = "/Users/Natsume/Downloads/DeepTrade_keras/dataset"
    for filename in os.listdir(dataset_dir):
        if filename != '000001.csv': # only use 000001.csv
           continue
        print("processing file: " + filename)
        filepath = dataset_dir + "/" + filename
        raw_data = read_sample_data(filepath) # list of objects (6434 days), each object is a day's stock info (date, OHLCV)
        moving_features, moving_labels = extract_feature(raw_data=raw_data, selector=selector, window=input_shape[0], with_label=True, flatten=True)
		# 1. 6464 days OHLCV, converted to array of 61 features, shape (61, 6464); 2. split (61, 6464) into a list of 6434 of (61, 30); 3. then convert it to array shape (6434, 61*30); but why not (6434, 61, 30)

		# save test_set and train_set
		# train_set: 5734 days, 1830 features (file), 1 target (file)
		# test_set: 700 days, 1830 features (file), 1 target (file)
        print("feature extraction done, start writing to file...")
        train_end_test_begin = moving_features.shape[0] - days_for_test
        if train_end_test_begin < 0:
            train_end_test_begin = 0
        for i in range(0, train_end_test_begin): # for each day
            for item in moving_features[i]: # for each feature
                fp.write("%s\t" % item) # separate each feature with tab
            fp.write("\n") # separate each day with new line
        for i in range(0, train_end_test_begin): # for each day
            lp.write("%s\n" % moving_labels[i]) # separate label with new line
        # test set: apply the same as training set above
        for i in range(train_end_test_begin, moving_features.shape[0]):
            for item in moving_features[i]:
                fpt.write("%s\t" % item)
            fpt.write("\n")
        for i in range(train_end_test_begin, moving_features.shape[0]):
            lpt.write("%s\n" % moving_labels[i])
		# only create dataset on 000001.csv not other files
        # break

    fp.close()
    lp.close()
    fpt.close()
    lpt.close()
