"""
Uses:
1. stock_csv_features_targets:
	- to turn a stock csv to features_array, target_array
2. example of it

Input:
1. file_path of csv

Return:
1. stock features_array
2. stock target_array
"""

# try dataset from csv to pandas to arrays
from prep_data_03_stock_01_csv_2_pandas_2_arrays_DOHLCV import csv_df_arrays

from prep_data_03_stock_02_OHLCV_arrays_2_features_targets_arrays import extract_feature

import numpy as np

def stock_csv_features_targets(filepath):

	# dates, opens, highs, lows, closes, volumes = csv_df_arrays(filepath)

	# all internal supported indicators are selected here
	user_indicators = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"] # for most model cases
	# user_indicators = ['MA'] # only used for qcg model

	# get features array and target array
	moving_indicators_features, moving_real_price_changes = extract_feature(selector=user_indicators, file_path=filepath)

	moving_indicators_features = np.transpose(moving_indicators_features, (0,2,1))
	return (moving_indicators_features, moving_real_price_changes)

"""
### example
mdjt_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices/mdjt_prices.csv"

mdjt_features, mdjt_target = stock_csv_features_targets(mdjt_path)
"""
