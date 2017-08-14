"""
csv_df_arrays
- convert csv to df
- convert df to many arrays
- from nan to number 0.0
- fill 0.0 with values before and after

Uses:
1. csv_df_arrays(file_path)
2. Example of using it

function input:
1. file_paths

Return:
dates, opens, highs, lows, closes, volumes
"""


import pandas as pd
import numpy as np

def csv_df_arrays(filename):
	# stock = pd.read_csv(filename).as_matrix()
	stock = pd.read_table(filename, sep=" ").as_matrix()

	dates = stock[:, 0].astype('<U19')
	opens = stock[:, 1].astype('float')
	highs = stock[:, 2].astype('float')
	lows = stock[:, 3].astype('float')
	closes = stock[:, 4].astype('float')
	volumes = stock[:, 5].astype('float')

	opens = np.nan_to_num(opens)
	highs = np.nan_to_num(highs)
	lows = np.nan_to_num(lows)
	closes = np.nan_to_num(closes)
	volumes = np.nan_to_num(volumes)

	def fill_zeros(values):
		for index in range(len(values)):
			if values[index] == 0.0: # handing trading date without trading info 停牌
				if values[index-1] != 0.0:
					values[index] = values[index-1]
				else:
					values[index] = values[index+1]
		return values

	opens = fill_zeros(opens)
	highs = fill_zeros(highs)
	lows = fill_zeros(lows)
	closes = fill_zeros(closes)
	volumes = fill_zeros(volumes)

	return (dates, opens, highs, lows, closes, volumes)




# Example

# stock_path = "/Users/Natsume/Downloads/data_for_all/stocks/commodities/A.txt"
#
# dates, opens, highs, lows, closes, volumes = csv_df_arrays(stock_path)
