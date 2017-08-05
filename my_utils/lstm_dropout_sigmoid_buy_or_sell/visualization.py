"""
plot_price_return_num_trades
- get predictions of a stock
- calc cumulative return
- calc num_trades
- do subplots

- reduce_trade_frequency
- continuous_color_reference
http://matplotlib.org/examples/color/colormaps_reference.html

Datasets: ETF 300 and ETF 50
- use the simple method to cut down trade frequency

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# 获取标的收盘价和日期序列
from stock_csv_pandas_array import csv_df_arrays
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index000001.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF50.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF500.csv"
index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF300.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index50.csv"

date,open_prices,_,_, closes, _ = csv_df_arrays(index_path)


from model_predict import get_stock_preds_target
index_preds_target = get_stock_preds_target(index_path)
"""
index_preds_target 中包含2组数据：
index_preds_target[:, 0]:预测值，即下一日的持仓的总资产占比
index_preds_target[:, 1]:下一日的当天价格变化
"""

#####################3
# plot with different time span
####################

# 30 days
# 90 days
time_span = 700  # 从今天回溯700 days
# time_span = 500  # 从今天回溯500 days
# time_span = 250  # 从今天回溯250 days
# time_span = 30  # 从今天回溯30 days
# time_span = 1  # 从今天回溯1 days
# from 20170720 to 20170728
# time_span = 11  # 昨天开始交易，到今天收盘，交易开始两天了



# zoom in and out for the last 700 trading days
open_prices = open_prices[-time_span:]
closes = closes[-time_span:]
index_preds_target = index_preds_target[-time_span:]
date = date[-time_span:]
y_pred = index_preds_target[:,0]
y_target = index_preds_target[:,1]
origin_y_pred = np.copy(y_pred)
origin_y_target = np.copy(y_target)
################################################################
# The latest algo to cut down trade frequency
################################################################
init_capital = 1000000 # 初始总现金
# y_pred = index_preds_target[:,0] # 预测值，当日持仓市值占比
count_1_99=0
count_99_more=0
count_01_less=0
for idx in y_pred:
	if idx >0.01 and idx < 0.99:
		count_1_99+=1
	elif idx>0.99:
		count_99_more+=1
	else:
		count_01_less+=1
print("total_prediction_count: %d, predictions over 0.99: %d, predictions below 0.01: %d; predictions in between: %d" % (y_pred.shape[0], count_99_more, count_01_less, count_1_99))
# print("original prediction before cutting frequency:", y_pred)
# origin_pred = np.copy(y_pred)
# y_true = index_preds_target[:,1] # 相邻两天收盘价变化率
daily_shares_pos = [] # 用于收集每日的持股数，让实际交易便捷
daily_capital = [] # 收集每日的总资产



buy_threshold=0.999 # 0.9 for ETF50, 0.99 for ETF 300
sell_threshold=0.001 # 0.1 for ETF50, 0.01 for ETF 300



# 加一个0.0而不是1.0在预测值序列的前面， 为什么呢？ 因为y_pred是预测第二天的价格变化率
y_pred = np.concatenate((np.array([0.0]), y_pred),0)
daily_shares_pos = [0.0] # 用于收集每日的持股数，让实际交易便捷
daily_capital = [init_capital]
daily_cash_left = [init_capital] # 收集当日的现金结余

# 将预测值（价格变化比）转变为涨跌预测，或者满仓或空仓预测
# 实际上交易是从第二天开始的，第一天没有对应的预测值
for idx in range(1, len(y_pred)-1):
	if y_pred[idx] > buy_threshold: # 大于买入阀值，才算买入，标记1.0
		y_pred[idx] = 1.0
	elif y_pred[idx] < sell_threshold: # 小于卖出阀值，才能卖出，0.0
		y_pred[idx] = 0.0
	else:
		y_pred[idx] = 0.5  # 等于阀值，全部是维持之前状态，0.5

# 实际上交易是从第二天开始的，第一天没有对应的预测值

shares_pos = 0.0 # 假设当天或者第二天的持股数为0.0
for idx in range(1, len(y_pred)-1):
	if idx == 1: # 实际交易都是从第二天开始的
		if y_pred[idx] == 1.0: # 如果第二天，当下y_pred是1.0
			shares_pos = np.trunc(1000000/open_prices[idx]/100)*100 # 全仓买入，100股为一手，计算所买入的股数
		elif y_pred[idx] == 0.0: # 如果第二天，当下是0.0；维持空仓
			shares_pos = 0.0
		else:
			shares_pos = 0.0  # 如果是0.5，维持空仓
		daily_shares_pos.append(shares_pos) # 收集第二天的持股数
		cost = shares_pos*open_prices[idx]*0.001 # 计算当天交易成本
		cash_left = 1000000 - shares_pos*open_prices[idx] - cost # 当天所剩现金
		daily_cash_left.append(cash_left) # 收集第一天现金结余额
		end_day_capital = shares_pos*closes[idx] + cash_left # 当天总资产
		daily_capital.append(end_day_capital) # 收集当天的总资产

	# 从第三天开始
	else:
		if y_pred[idx-1] == 1.0: # 如果昨天的预测值y_pred是1.0
			if y_pred[idx] == 1.0: # 如果用于当天交易的预测值是1.0，维持仓位
				shares_pos = daily_shares_pos[idx-1]
			elif y_pred[idx] == 0.0: # 如果用于当天交易的预测值是0.0，那么全仓卖出
				shares_pos = 0.0 # 持股数为0
			else: # 如果用于当天交易的预测值是0.5，那么维持昨天状态，维持昨天持股数
				shares_pos = daily_shares_pos[idx-1]


		elif y_pred[idx-1] == 0.0: # 如果昨天的预测值y_pred是0.0
			if y_pred[idx] == 1.0: # 如果用于当天交易的预测值是1.0，全仓买入
				shares_pos = np.trunc(daily_capital[idx-1]/open_prices[idx]/100)*100
			elif y_pred[idx] == 0.0: # 如果用于当天交易的预测值是0.0，那么维持空仓
				shares_pos = 0.0
			else:
				shares_pos = 0.0 # 如果用于当天交易的预测值是0.5，那么维持昨天持仓


		elif y_pred[idx-1] == 0.5 and daily_shares_pos[idx-1]>0.0: # 如果昨天的预测值y_pred是0.5,同时昨天的持仓是满仓 （非0持股，就是满仓）
			if y_pred[idx] == 1.0: # 如果用于当天交易的预测值是1.0，维持仓位
				shares_pos = daily_shares_pos[idx-1]
			elif y_pred[idx] == 0.0: # 如果用于当天交易的预测值是0.0，那么全仓卖出
				shares_pos = 0.0
			else: # 如果用于当天交易的预测值是0.5， 那么维持昨天的持股数
				shares_pos = daily_shares_pos[idx-1]


		elif y_pred[idx-1] == 0.5 and daily_shares_pos[idx-1] == 0.0: # 如果昨天的预测值y_pred是0.5,同时昨天的持仓是空仓 （非0持股，就是满仓）
			if y_pred[idx] == 1.0: # 如果用于当天交易的预测值是1.0，满仓买入
				shares_pos = np.trunc(daily_capital[idx-1]/open_prices[idx]/100)*100
			elif y_pred[idx] == 0.0: # 如果用于当天交易的预测值是0.0，维持空仓
				shares_pos = 0.0
			else: # 如果用于当天交易的预测值是0.5， 那么维持昨天的持股数，0.0
				shares_pos = 0.0

		daily_shares_pos.append(shares_pos) # 收集从第三天开始的总持仓
		cost = np.abs(np.array(shares_pos) - np.array(daily_shares_pos[idx-1]))*open_prices[idx]*0.001 # 当天交易成本
		if shares_pos == daily_shares_pos[idx-1]: # 如果今天的持股数 == 昨天持股数
			cash_left = daily_cash_left[idx-1] # 没有交易和交易成本，昨天的现金剩余==今天的现金剩余
		else: # 如果持股数发生了变化，那么有交易，今天的现金剩余要重新计算
			cash_left = daily_capital[idx-1] - shares_pos*open_prices[idx] - cost # 当天的现金结余
		daily_cash_left.append(cash_left)
		end_day_capital = shares_pos*closes[idx] + cash_left # 当天总资产
		daily_capital.append(end_day_capital) # 收集当天总资产

# 计算累积总资金曲线
accum_profit = (np.array(daily_capital)/daily_capital[0])-1# 累积总资金曲线， 减去初始资金 1，获得收益累积曲线
print("time span: from %s to %s" % (date[-time_span], date[-1])) # 最近日期
print("last open price:", open_prices[-1])
print("last close price:", closes[-1])
print("final_return:", accum_profit[-1]) # 累积总收益
print("predictions: ", y_pred[-3:])
print("shares_pos: ", daily_shares_pos[-1])
print("capitals: ", daily_capital[-1])

############################################################
# replace_value_with_earlier_value
# 将持股数统计中的0.0，用紧邻的持股数代替
############################################################
# 让持股为0的交易日的持股数转化为前一天的持股数或后一天的持股数
daily_shares_pos_non0 = np.copy(daily_shares_pos)
# 将前一天的持股数来取代后一天的0.0 股数
for idx in range(1, len(daily_shares_pos_non0)):
	if daily_shares_pos_non0[idx] == 0.0:
		daily_shares_pos_non0[idx] = daily_shares_pos_non0[idx-1]
# 用后一天的持股数，来取代前一天的0
for idx in range(len(daily_shares_pos_non0)-2, -1, -1):
	if daily_shares_pos_non0[idx] == 0.0:
		daily_shares_pos_non0[idx] = daily_shares_pos_non0[idx+1]
print("all non_0 shares_pos is done")

############################################################
# 买入持有，不要频繁交易，在2014年至2015上半年，是最优策略；
# 频繁买卖，会损失很多盈利
# view_all_outputs_together
############################################################
origin_pred = np.copy(origin_y_pred)
display_dataset = np.concatenate((np.array(origin_pred).reshape((-1,1)), np.array(y_pred).reshape((-1,1))[:-1], np.array(daily_shares_pos).reshape((-1,1)), np.array(daily_shares_pos_non0).reshape((-1,1)), np.array(daily_capital).reshape((-1,1)), np.array(open_prices).reshape((-1,1)), np.array(date).reshape((-1,1))), axis=1)
import pandas as pd
display_pd = pd.DataFrame(display_dataset)
display_pd.columns = ['origin_pred', 'y_pred', 'daily_shares_pos', 'daily_shares_pos_non0', 'daily_capital', 'open_prices', 'date']
print("display_dataset", display_pd.head(100))

############################################################
# How to calculate 换手率 turnover_rate
############################################################
# the first value of this list is ignored
changes_pos_rate = np.abs(np.array(daily_shares_pos[1:]) - np.array(daily_shares_pos[:-1]))/np.array(daily_shares_pos_non0[:-1]) # 相邻两日的持仓占比之差（变化）
# we don't care whether changes_pos_rate[idx] is how much away from 1.0
for idx in range(len(changes_pos_rate)):
	if changes_pos_rate[idx] > 0.0:
		changes_pos_rate[idx] = 1.0

turnover_rate = np.cumsum(changes_pos_rate) # 累积差值，获得总资产进出市场的次数


############################################################
# calc 胜率 winning_ratio
############################################################
#####changes_pos_rate: 0 为持仓或者空仓，1为买或卖；
### num_trade_actions： 累积卖和买的总次数
num_trade_actions = np.sum(changes_pos_rate)
################### how many winning trades = how many times current_trade_end_capital is greater than previous_trade_end_capital
### trades_record： 记录哪天发生过交易（不论是买入还是卖出）
trades_record = np.copy([0.0] + changes_pos_rate.tolist())
# find the index of every close_position date
count_trade = 0
for idx in range(1, len(trades_record)):
	# 如果当天交易信号是1.0， 记录一次交易信号
	if trades_record[idx] == 1.0:
		count_trade+=1
	# 如果交易信号的次数是2的整数倍，且当天也发生了一次交易
	if count_trade % 2 == 0.0 and trades_record[idx] == 1.0:
		# 那么将这一天的交易信号标记为2.0，作为平仓交易信号
		trades_record[idx] = 2.0
# 收集平仓交易当天的收盘总资产
close_pos_capital = np.array(daily_capital)[trades_record == 2.0]
# 收集开仓交易当天收盘的总资产
close_pos_capital_0 = np.array(daily_capital)[trades_record == 1.0]
# 收集开仓交易前一日收盘的总资产
open_pos_capital=[]
for idx in range(len(trades_record)):
	if trades_record[idx] == 1.0:
		index = idx-1
		open_pos_capital.append(daily_capital[index])
open_pos_capital_arr = np.array(open_pos_capital)
# 每笔交易（已完结的交易）的盈亏额度
closed_trades_profit_loss = close_pos_capital - open_pos_capital_arr[:-1]
#### num_full_trades： 已经平仓的交易次数
num_full_trades = closed_trades_profit_loss.shape[0]
# compare one close_pos_capital with another to find the winning trades
winning_trades_sum = (closed_trades_profit_loss>0).sum()
losing_trade_sum = (closed_trades_profit_loss<0).sum()
winning_rate = winning_trades_sum/num_full_trades
print("winning rate: ", winning_rate)

############################################################
# 盈亏比
############################################################
total_profit = 0.0
total_loss = 0.0
for profit_loss in closed_trades_profit_loss:
	if profit_loss > 0.0:
		total_profit+=profit_loss
	else:
		total_loss-=profit_loss
avg_profit_trade = total_profit/winning_trades_sum
avg_loss_trade = total_loss/losing_trade_sum
avg_profit_loss_rate = avg_profit_trade/avg_loss_trade

############################################################
# 截止最近一天的净盈利总额
# 已完结交易的净盈利+未完结交易的净盈利
############################################################
# 已完结交易的净盈利
closed_net_profit = closed_trades_profit_loss.sum()
open_net_profit = daily_capital[-1] - open_pos_capital_arr[-1]
total_net_profit = closed_net_profit + open_net_profit
total_profit_rate = total_net_profit/daily_capital[0]



############################################################
# create an array to display winning trades
# 胜：红色bar；负：绿色bar
# full_trades_positions, winning_trades_positions
############################################################
# 让所有平仓交易日的信号值为1.0； 其他所有日信号为0.0
# 平仓交易日所在
close_trades_positions = np.zeros(len(trades_record))
# 开仓交易日所在
open_trades_positions = np.zeros(len(trades_record))
for idx in range(len(trades_record)):
	if trades_record[idx] == 2.0:
		close_trades_positions[idx] = 1.0

	if trades_record[idx] == 1.0:
		open_trades_positions[idx] = 1.0
close_trades_positions.sum()
open_trades_positions.sum()

# get full_trades_capital only, not the full capital dataset
# full_trades_capital = np.array(daily_capital)[np.array(full_trades_positions) == 1.0]
# check to see whether the first element is 1 or not
# full_trade_idx = 0
# winning_trades_positions = np.copy(full_trades_positions)
# for idx in range(len(winning_trades_positions)):
# 	if winning_trades_positions[idx] == 1.0:
# 		if full_trade_idx == 0:
# 			if full_trades_capital[full_trade_idx] > 1000000:
# 				winning_trades_positions[idx] = 1.0
# 			else:
# 				winning_trades_positions[idx] = 0.0
#
# 		else:
# 			if full_trades_capital[full_trade_idx] > full_trades_capital[full_trade_idx-1]:
# 				winning_trades_positions[idx] = 1.0
# 			else:
# 				winning_trades_positions[idx] = 0.0
# 		full_trade_idx+=1

############################################################
# 记录历史回撤  record_drawdown maximum_drawdown
# drawdown: 今天总资产小于昨天总资金，今天出现回撤
# maximum_drawdown: 最大回撤，是今天总资产相对于之前最高总资产，下降的幅度
############################################################
# record the highest capital so far 记录截止当下最高总资产
highest_capital_so_far = [1000000]
for idx in range(1,len(daily_capital)):
	if daily_capital[idx] > daily_capital[idx-1]:
		if daily_capital[idx] > highest_capital_so_far[idx-1]:
			highest_capital_so_far.append(daily_capital[idx])
		else:
			highest_capital_so_far.append(highest_capital_so_far[idx-1])
	else:
		highest_capital_so_far.append(highest_capital_so_far[idx-1])

# 每天的回撤记录
capital_drawdown=[]
capital_drawdown_rate=[]
for idx in range(len(daily_capital)):
	# 有回撤 >0, 无回撤 == 0
	capital_drawdown.append(highest_capital_so_far[idx]-daily_capital[idx])
	capital_drawdown_rate.append((highest_capital_so_far[idx]-daily_capital[idx])/highest_capital_so_far[idx])
maximum_drawdown = np.array(capital_drawdown).max()
maximum_drawdown_rate = np.array(capital_drawdown_rate).max()
#### let's plot drawdown rate history curve
#



################################################################
#### train with validation 700 dataset
################################################################
###### what is the frequencies of y_pred
y_pred_hist = np.copy(origin_y_pred)
y_target_hist = np.copy(origin_y_target)

###### what is the accuracy of predicting up or down
for idx in range(len(origin_y_target)):
	if origin_y_target[idx] > 0.0:
		origin_y_target[idx] = 1.0
	else:
		origin_y_target[idx] = 0.0

for idx in range(len(origin_y_pred)):
	if origin_y_pred[idx] > 0.0:
		origin_y_pred[idx] = 1.0
	else:
		origin_y_pred[idx] = 0.0

accuracy = (np.array(origin_y_pred[:-1]) == np.array(origin_y_target[1:])).sum()/len(np.array(origin_y_target[1:]))
print("accuracy of guessing up and down:", accuracy)

################################################################
# 将预测值融入价格曲线 # prediction_as_price_curve_color
################################################################



## color data for close prices： 经过阀值调节过的预测值

color_data = y_pred


## 用于画收盘价曲线的数据
target_closes = closes/closes[0]

# different ways of setting color data

line_data = target_closes.reshape((-1,1))-1 # close_price normalized to start from 1 onward, so minus 1 to bring it down to 0 original for plotting

def uniqueish_color(color_data):
    """There're better ways to generate unique colors, but this isn't awful."""
    # return plt.cm.gist_ncar(color_data)
    # return plt.cm.binary(color_data)
    return plt.cm.bwr(color_data)

X = np.arange(len(line_data)).reshape((-1,1))
y = line_data
xy = np.concatenate((X,y), axis=1)



plt.figure()
ax1 = plt.subplot2grid((14, 3), (0, 0), colspan=3, rowspan=4)
#############
### plot close_price curve and fill predictions as continuous color ######
#############
for start, stop, col in zip(xy[:-1], xy[1:], color_data):
    x, y = zip(start, stop)
    ax1.plot(x, y, color=uniqueish_color(col))
    ax1.set_title('predict up or down, accuracy: %02f' % accuracy)

ax1.plot(accum_profit, c='gray', alpha=0.5, label='accum_profit')
ax1.legend(loc='best')
ax1.set_title('ETF300 from %s to %s return: %04f' % (date[0], date[-1], accum_profit[-1]))

ax2 = plt.subplot2grid((14, 3), (4, 0), colspan=3, rowspan=2)
ax2.hist(y_pred_hist, color='red', alpha = 0.5, label='pred')
ax2.hist(y_target_hist, color='blue', alpha = 0.2, label='target')
ax2.legend(loc='best')
ax2.set_title("y_pred_target frequencies")



#############
### drawdown curve
#############
ax3 = plt.subplot2grid((14, 3), (6, 0), colspan=3, rowspan=2)
ax3.plot(capital_drawdown_rate, c='gray', label='drawdown_curve')
ax3.legend(loc='best')
ax3.set_title("maximum drawdown: rate: %02f, capital: %d" % (maximum_drawdown_rate, maximum_drawdown))

##########################
#### 画出开仓和平仓所在的位置
##########################
ax4 = plt.subplot2grid((14, 3), (8, 0), colspan=3, rowspan=2)
X = np.arange(len(close_trades_positions))
ax4.bar(X, open_trades_positions, facecolor='gray', edgecolor='gray')
ax4.bar(X, close_trades_positions, facecolor='pink', edgecolor='pink')

ax4.set_title('open position (gray), close position(pink), %d winning trades, %d trades in total' % (winning_trades_sum, num_full_trades)) # change model name


#############
### plot 换手率,
#############
ax5 = plt.subplot2grid((14, 3), (10, 0), colspan=3, rowspan=2)
ax5.plot(turnover_rate, c='red', label='turnover_rate')
ax5.legend(loc='best')
ax5.set_title("TurnOver Rate: %d, avg_profits/avg_losses: %02f, total_net_profit/init_capital: %02f" % (turnover_rate[-1], avg_profit_loss_rate, total_profit_rate))
#
#
# ### plot daily_shares_pos curve
ax6 = plt.subplot2grid((14, 3), (12, 0), colspan=3, rowspan=2)
init_shares_full = init_capital/open_prices[0]
daily_shares_rate = np.array(daily_shares_pos)/init_shares_full
ax6.plot(daily_shares_rate, c='k', label='daily_shares_rate')
ax6.legend(loc='best')
ax6.set_title('init_share_number: %d, latest_share_number: %d' % (init_shares_full, daily_shares_pos[-1])) # change model name






plt.tight_layout()
plt.show()



# plt.savefig("/Users/Natsume/Downloads/data_for_all/stocks/model_performance/ETF300_model4_addcost.png")
