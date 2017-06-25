"""
### Summary
- combine training, valid, test predictions arrays into one array
- get cumulative return curve
- do subplot close price, return curve and positions bar
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# import OHLCV arrays
from prep_data_03_stock_01_csv_2_objects_2_arrays_DOHLCV import closes

# import train_pos_targets, valid_pos_targets, test_pos_targets
from prep_data_utils_01_save_load_large_arrays_bcolz_np_pickle_torch import bz_load_array
train_pos_targets_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/train_pos_targets"
valid_pos_targets_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/valid_pos_targets"
test_pos_targets_path = "/Users/Natsume/Downloads/DeepTrade_keras/features_targets_data/test_pos_targets"
train_pos_targets = bz_load_array(train_pos_targets_path)
valid_pos_targets = bz_load_array(valid_pos_targets_path)
test_pos_targets = bz_load_array(test_pos_targets_path)

# there are 2 csv files used to build dataset
num_csv = 2
if num_csv > 0:
	# we will only use 000001.csv data, so we slice the arrays above by half
	train_pos_targets = train_pos_targets[0: int(len(train_pos_targets)/num_csv)]
	valid_pos_targets = valid_pos_targets[0: int(len(valid_pos_targets)/num_csv)]
	test_pos_targets = test_pos_targets[0: int(len(test_pos_targets)/num_csv)]


full_pos_targets = np.concatenate((train_pos_targets, valid_pos_targets, test_pos_targets), axis=0)
target_pos_price_change = full_pos_targets[-1500:]
target_closes = closes[-1500:]
# train_pos_targets[0]: array([ 0.62353933,  0.00433955]): next_day_pos, today's price_change_pct
# next_day_profit =
daily_profit = [0]

for idx in range(len(target_pos_price_change)):
	# 1: start with 1 share of stock at price 1
	# train_pos_targets[idx,0]: how many shares to keep for next day
	# train_pos_targets[idx+1,1]: price change pct of next day
	# following code gives us everyday's profit
	daily_profit.append(1*target_pos_price_change[idx,0]*target_pos_price_change[idx,1])
	if idx+1 == len(target_pos_price_change)-1:
		break

# accumulated profit
accum_profit = np.cumsum(np.array(daily_profit))

plt.figure()
ax1 = plt.subplot2grid((5, 3), (0, 0), colspan=3, rowspan=3)  # stands for axes
ax1.plot(target_closes, c='blue', label='close_price')
ax1.set_title('close_prices')
ax2 = plt.subplot2grid((5, 3), (3, 0), colspan=3)
ax2.plot(accum_profit, c='red', label='train_profit')
ax3 = plt.subplot2grid((5, 3), (4, 0), colspan=3)
X = np.arange(len(full_pos_targets))
ax3.bar(X, full_pos_targets[:,0], facecolor='#9999ff', edgecolor='blue')
# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax4.scatter([1, 2], [2, 2])
# ax4.set_xlabel('ax4_x')
# ax4.set_ylabel('ax4_y')
# ax5 = plt.subplot2grid((3, 3), (2, 1))

plt.tight_layout()
plt.show()
