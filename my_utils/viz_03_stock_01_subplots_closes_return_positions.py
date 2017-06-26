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

train_pos_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/positions_priceChanges/train_pos_priceChange"
valid_pos_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/positions_priceChanges/valid_pos_priceChange"
test_pos_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/positions_priceChanges/test_pos_priceChange"

train_pos_targets = bz_load_array(train_pos_targets_path)
valid_pos_targets = bz_load_array(valid_pos_targets_path)
test_pos_targets = bz_load_array(test_pos_targets_path)

# there are 2 csv files used to build dataset
num_csv = 1
if num_csv > 0:
	# we will only use 000001.csv data, so we slice the arrays above by half
	train_pos_targets = train_pos_targets[0: int(len(train_pos_targets)/num_csv)]
	valid_pos_targets = valid_pos_targets[0: int(len(valid_pos_targets)/num_csv)]
	test_pos_targets = test_pos_targets[0: int(len(test_pos_targets)/num_csv)]

# rbind train, valid and test pos_preds and price_change
full_pos_targets = np.concatenate((train_pos_targets, valid_pos_targets, test_pos_targets), axis=0)
target_pos_price_change = full_pos_targets[-700:]
target_closes = closes[-700:]

daily_capital1 = []
daily_capital2 = []
target_closes = target_closes/target_closes[0]# normalized price

### daily_capital == previous_day_capital + today_profit == previous_day_capital + today_capital*today_price_change == previous_day_capital + previous_day_capital*how_much_hold_for_today*today_price_change == previous_day_capital(1 + how_much_hold_for_today*today_price_change)

# before the first day, we opened position at the end of the day, no profit made, just original captial, we set it to be 1
# now for first day
daily_capital1.append(1)
daily_capital2.append(1)
## calc second_day capital onward
# target_pos_price_change[0][0] == how much capital hold for second day
# target_pos_price_change[1][1] == price change pct for second day
# second day capital == 1 * (1 + target_pos_price_change[0][0] * target_pos_price_change[1][1])

for idx in range(1, len(target_pos_price_change)):
	daily_capital1.append(daily_capital1[idx-1]*(1+target_pos_price_change[idx-1,0]*target_pos_price_change[idx,1]))
accum_profit1 = np.array(daily_capital1)-1

for idx in range(1, len(target_pos_price_change)):
	daily_capital2.append(daily_capital2[idx-1]*(1+target_pos_price_change[idx,0]*target_pos_price_change[idx,1]))
accum_profit2 = np.array(daily_capital2)-1
# # perform worse, but probably correct way to get profit curve
# daily_profit=[0] # day1 capital = 1
# for idx in range(1, len(target_pos_price_change)):
# 	daily_profit.append(target_closes[idx-1]*target_pos_price_change[idx-1,0]*target_pos_price_change[idx,1])
# accum_profit1 = np.cumsum(daily_profit)
#
# # perform better, but probably get days wrong
# daily_profit=[0] # day1 capital = 1
# for idx in range(1, len(target_pos_price_change)):
# 	daily_profit.append(target_closes[idx]*target_pos_price_change[idx,0]*target_pos_price_change[idx,1])
# accum_profit2 = np.cumsum(daily_profit)

# todo: plot the max and latest value on graph
plt.figure()
ax1 = plt.subplot2grid((7, 3), (0, 0), colspan=3, rowspan=3)  # stands for axes
ax1.plot(target_closes, c='blue', label='close_price')
ax1.set_title('close_prices')
ax2 = plt.subplot2grid((7, 3), (3, 0), colspan=3, rowspan=3)
ax2.plot(accum_profit2, c='red', label='train_profit')
ax3 = plt.subplot2grid((7, 3), (6, 0), colspan=3)
X = np.arange(len(target_pos_price_change))
ax3.bar(X, target_pos_price_change[:,0], facecolor='#9999ff', edgecolor='blue')


plt.tight_layout()
plt.show()
