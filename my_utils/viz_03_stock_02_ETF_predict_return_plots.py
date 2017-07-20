"""
plot_price_return_num_trades
- get predictions of a stock
- calc cumulative return
- calc num_trades
- do subplots

- continuous_color_reference
http://matplotlib.org/examples/color/colormaps_reference.html

### Summary
- combine training, valid, test predictions arrays into one array
- get cumulative return curve
- do subplot close price, return curve and positions bar
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# 获取标的收盘价和日期序列
from prep_data_03_stock_01_csv_2_pandas_2_arrays_DOHLCV import csv_df_arrays
index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index000001.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF50.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF500.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF300.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index50.csv"

date,open_prices,_,_, closes, _ = csv_df_arrays(index_path)


from predict_model_03_stock_01_predict_stock_best_model_save_result import get_stock_preds_target
index_preds_target = get_stock_preds_target(index_path)
"""
index_preds_target 中包含2组数据：
index_preds_target[:, 0]:预测值，即下一日的持仓的总资产占比
index_preds_target[:, 1]:下一日的当天价格变化
"""



# zoom in and out for the last 700 trading days
open_prices = open_prices[-700:]
closes = closes[-700:]
index_preds_target = index_preds_target[-700:]

### get return curve dataset
target_closes = closes/closes[0]# normalized price to price changes
open_prices = open_prices/open_prices[0]# normalized price to price changes
daily_capital=[]

######################################################
# 收益计算 _ v1.0, 对我的脑力要求太高，can't do it!
######################################################
	# 设置阀值，筛选买入，持有和卖出的时间点
	# 买入： 在第n个（pos>0.95）出现时，买入; 更改持仓状态
	# 持有： 0.05 < pos < 0.95时，均持有;
	# 卖出： 在第n个 （pos < 0.05）出现时，卖出; 更改持仓状态
	# 持仓状态： has_pos = True or False
# buy_signal = 0.9
# sell_signal = 0.1
# patience = 3 # 1,2,3
# pos = index_preds_target[:,0]
# price_ch = index_preds_target[:,1]
# pos_state = False
# buy_signal_count = 1
# sell_signal_count = 1
# init_capital = 100000
# daily_capital_list = []
#
# ### 搜索最佳组合
# for (buy_signal, sell_signal) in [(0.9,0.1), (0.8, 0.2), (0.7, 0.3)]:
# 	for patience in [1,2,3]:
#
# 		daily_capital=[]
# 		for index in range(len(pos)):
# 			# 买入条件： 没有持仓，买入信号出现 buy_signal_count 次
# 			if pos[index] > buy_signal and buy_signal_count == patience and pos_state == False:
# 				pos_state = True # 设置持仓状态
# 				if len(daily_capital) == 0:
# 					daily_capital.append(init_capital * (1 + price_ch[index])) # 计算当日收益，收集
# 				else:
# 					daily_capital.append(daily_capital[index-1] * (1 + price_ch[index])) # 计算当日收益，收集
# 				buy_signal_count += 1
# 				sell_signal_count = 1
#
# 			# 卖出条件： 有持仓，卖出信号出现 sell_signal_count 次
# 			elif pos[index] < sell_signal and buy_signal_count == patience and pos_state == True:
# 				pos_state = False # 设置持仓状态
# 				sell_signal_count+=1
# 				buy_signal_count = 1
# 				daily_capital.append(daily_capital[index-1]) # 计算当日收益，收集
#
#
# 			else:
# 				# 持有状态下, 计算当天市值，收集
# 				if pos_state == True and index > 0:
# 					daily_capital.append(daily_capital[index-1] * (1 + price_ch[index]))
# 				# 空仓状态下，计算当天市值，收集
# 				else:
# 					if len(daily_capital) == 0:
# 						daily_capital.append(init_capital)
# 					else:
# 						daily_capital.append(daily_capital[index-1])
# 				# 依旧累积买入和卖出信号发生次数
# 				if pos[index] > buy_signal:
# 					buy_signal_count+=1
# 				if pos[index] < sell_signal:
# 					sell_signal_count+=1
#
# 		daily_capital_list.append(np.array(daily_capital))

# for captial in daily_capital_list:
# 	plt.plot(capital)
# plt.show()

######################################################
# 最早期设计的总资产和换手率计算方法
######################################################
"""
### 如何计算累积的每日的总资产（包括计算交易成本）？代码在下面可见

daily_capital.append(1. * (1. + index_preds_target[0,0]*index_preds_target[0,1]))
- 上面第一个 '1' : 初始本金
- 1 * index_preds_target[0,0]*index_preds_target[0,1]: 持仓市值产生的利润 = 1 * 当日开盘时所持仓资产占总资产的比例（即前一日收盘时作出的预测值）* 当日价格的变化值

accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
daily_capital.append(accum_capital)
- daily_capital[idx-1]: 昨天的总资产
- daily_capital[idx-1]*index_preds_target[idx,0]*index_preds_target[idx,1]:今天持仓市值产生的利润 = 昨日收盘总资产 * 当日开盘时所持仓资产占总资产的比例（即昨日收盘时作出的预测值）* 当日价格的变化值

daily_capital.append(accum_capital): 将累积的每天的总资产收入daily_capital这个list中

### 如何在计算每日总资产的同时，减去交易成本？
if index_preds_target[idx-1,0] == index_preds_target[idx,0] == 1.0 or index_preds_target[idx-1,0] == index_preds_target[idx,0] == 0.0:
- 如果昨天的持仓占比于今天的持仓占比都是100%， 那么不存在交易，也就没有交易成本；
- 如果昨天的持仓占比于今天的持仓占比都是0%， 那么不存在交易，也就没有交易成本；

elif index_preds_target[idx-1,0] == index_preds_target[idx,0] and index_preds_target[idx-1,1] == index_preds_target[idx,1] == 0.0:
- 如果昨天的持仓占比 == 今天的持仓占比， 而且昨日价格变化和今日价格变化都是0， 那么没有交易，也就没有交易成本

else:
	# cost = (today's total_share_number_to_hold - yesterday_total_share_number_to_hold)*Today_open_price*0.001
	cost = np.abs((daily_capital[idx-1]*index_preds_target[idx,0]- daily_capital[idx-2]*index_preds_target[idx-1,0])*0.001)
	# today's accum_capital = today's accum_capital - cost
	accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1]) - cost
	daily_capital.append(accum_capital)

那么其他情况，全部存在交易成本问题：

cost = np.abs((daily_capital[idx-1]*index_preds_target[idx,0]- daily_capital[idx-2]*index_preds_target[idx-1,0])*0.001)
- 今日交易成本 = abs(今日持仓资产 - 昨日持仓资产）* 交易成本占比 = abs(昨日总资产*今日持仓的资产占比 - 前日总资产*昨日持仓的资产占比)*交易成本占比

accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1]) - cost
- 今日总资产 = 昨日总资产 + 昨日总资产*今日持仓的总资产占比*今日价格变化 - 今日交易成本

daily_capital.append(accum_capital)
- 将累积的每天的总资产收入daily_capital这个list中
"""

#
# ## see the picture the author drew to refresh the logic
#
# # from second day onward
# for idx in range(len(index_preds_target)):
#
# 	# 第一天的市值： 第一天开盘买入，当天结束时的市值
# 	if idx == 0:
# 		daily_capital.append(1. * (1. + index_preds_target[0,0]*index_preds_target[0,1]))
#
# 	else:
# 		# # 情况1:
# 		# # 完全忽略交易成本
# 		# accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
# 		# daily_capital.append(accum_capital)
#
# 		# 情况2：
# 		# 计算实际交易成本
# 		if index_preds_target[idx-1,0] == index_preds_target[idx,0] == 1.0 or index_preds_target[idx-1,0] == index_preds_target[idx,0] == 0.0:
# 			# no trade, no trading cost today
# 			accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
# 			daily_capital.append(accum_capital)
# 		elif index_preds_target[idx-1,0] == index_preds_target[idx,0] and index_preds_target[idx-1,1] == index_preds_target[idx,1] == 0.0:
# 			# no trade, no trading cost today
# 			accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
# 			daily_capital.append(accum_capital)
#
# 		else:
# 			# cost = (today's holding position capital - yesterday's holding position capital)*0.001
# 			cost = np.abs((daily_capital[idx-1]*index_preds_target[idx,0]- daily_capital[idx-2]*index_preds_target[idx-1,0])*0.001)
# 			# today's accum_capital = today's accum_capital - cost
# 			accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1]) - cost
# 			daily_capital.append(accum_capital)
#
# ### 统计总收益
# accum_profit = np.array(daily_capital)-1 # 累积总资产减去初始资产 = 累积收益
# print("final date:", date[-1])
# print("final_return:", accum_profit[-1])
#
# ##### accumulation of transaction percentage，即换手率
# # 这里我们用市值占比率，从0到1， 从1到0， 来理解计算换手率
# preds = index_preds_target[:,0] # 每日持仓的总资产占比
# changes_preds = np.abs(preds[1:] - preds[:-1]) # 相邻两日的持仓占比之差（变化）
# turnover_rate = np.cumsum(changes_preds) # 累积差值，获得总资产进出市场的次数

######################################################
# 降频方案：将预测值看作持股数量占比（全仓持股总数固定不变），不再是市值在总资产中占比
# 基于此方案的总资产和换手率计算方法
# understand_y_pred_differently_solve_trade_frequency
#####################################################

# 原理： 从最简单的变化入手，现有模型上修改，现有交易理解上改进
# 改进预测值理解： 不是市值占比，而是持股数占比
# 持股总数不变，可以提前设置，比如总数设置为10000股； 预测总股数比率， 80% 就是8000股；
# 设置：总投资额 = 1； 总持股数 = 1
# 预测值 or index_preds_target[0,0] == 预测当日开盘后需要的持股数占比
init_capital = 1.0
daily_capital = []
daily_trade_pct = []
add_cost = True
threshold = True

# 第一天的交易股数占比：当前预测值， index_preds_target[0,0]
# 第一天的市值： 第一天开盘买入，当天结束时的市值 = 当天总资产（比如，1） + 当天持仓市值在收盘时的增值部分 = 当天总资产  + 当天开盘后形成的持股数占比 * 当天开盘价 * 当天价格变化比 = 1 + index_preds_target[0,0] * open_prices[0] * index_preds_target[0,1]
day_one_capital = 1 + index_preds_target[0,0] * open_prices[0] * index_preds_target[0,1]
# 第一天的交易成本： 当天开盘后形成的持股数占比 * 当天开盘价 * 0.001 = index_preds_target[0,0] * open_prices[0] * 0.001
day_one_trade_pct = index_preds_target[0,0]
day_one_cost = day_one_trade_pct * open_prices[0] * 0.001
# 第一天收盘实际市值： 第一天的市值 - 第一天的交易成本
day_one_capital_add_cost = day_one_capital - day_one_cost
# 收集每日交易股数占比
daily_trade_pct.append(day_one_trade_pct)
# 收集
if add_cost:
	daily_capital.append(day_one_capital_add_cost)

else:
	daily_capital.append(day_one_capital)


# 第二天及以后的变化：
for idx in range(1, len(index_preds_target)):

# 第二天的交易股数占比： 当前预测值 - 昨天预测值 = index_preds_target[idx,0] - index_preds_target[idx-1,0]
	day_two_trade_pct = np.abs(index_preds_target[idx,0] - index_preds_target[idx-1,0])

############################
# threshold_reduce_noise_trade
# 第二天及以后的变化 + 阀值控制
# 第二天的交易股数占比： 当前预测值 - 昨天预测值
# 阀值： 如果， 第二天的交易股数占比 < 0.1; 那么，当前预测值 = 昨天预测值; 即，不做交易，继续持有不变
	if threshold and day_two_trade_pct < 0.85:
		day_two_trade_pct = 0.0
		index_preds_target[idx,0] = index_preds_target[idx-1,0]

# 第二天的市值： 第二天开盘买入，当天结束时的市值 = 昨天总资产 + 当天持仓市值在收盘时的增值部分 = 昨天收盘总资产  + 当天开盘后形成的持股数占比 * 当天开盘价 * 当天价格变化 = 昨天收盘总资产 + index_preds_target[idx,0] * open_prices[idx] * index_preds_target[idx,1]
	day_two_capital = daily_capital[idx-1] + index_preds_target[idx,0] * open_prices[idx] * index_preds_target[idx,1]


# 第二天的交易成本： 第二天的交易股数占比 * 0.001 * 第二天的开盘价
	day_two_cost = day_two_trade_pct * open_prices[idx] * 0.001
# 第二天收盘实际市值： 第二天的市值 - 第二天的交易成本
	day_two_capital_add_cost = day_two_capital - day_two_cost

# 收集每日交易股数占比
	daily_trade_pct.append(day_two_trade_pct)

# 收集
	if add_cost:
		daily_capital.append(day_two_capital_add_cost)

	else:
		daily_capital.append(day_two_capital)

### 统计总收益
accum_profit = np.array(daily_capital)-1 # 累积总资产减去初始资产 = 累积收益
print("final date:", date[-1])
print("final_return:", accum_profit[-1])

# 如何计算换手率？
# 所以第二天交易股数占比总和
turnover_rate = np.cumsum(np.array(daily_trade_pct))

######################################################
# revised trading cost based on MonteCarlo's suggestion
#####################################################
# add_cost = True # False # True
# threshold = True # False # True
# daily_stock_shares = []
# daily_capital = []
#
# for idx in range(len(index_preds_target)):
#
# 	if idx == 0:
# 		# 第一天的市值： 第一天开盘买入，当天结束时的市值 = 当天总资产 + 当天持仓市值在收盘时的增值部分 = 当天总资产（1 + 持仓市值占总资产比 * 当天价格变化）
# 		# 设定，初始资金=1，
# 		today_capital_before_cost = 1. * (1. + index_preds_target[0,0]*index_preds_target[0,1])
#
# 		# 第一天的持股数量： 第一天开盘时要用于持仓的总金额／第一天开盘价
# 		daily_stock_shares.append(1*index_preds_target[0,0]/open_prices[idx]) # open_prices normalized to start from 1, so that avoid 1/0 situation
#
# 		# 如果需要计算交易成本
# 		if add_cost:
# 			# 第一天的交易成本 = 首日买入持有市值*交易成本比率
# 			today_cost = 1 * index_preds_target[0,0] * 0.001
# 			# 第一天的市值 = 未计算交易成本的市值 - 交易成本
# 			today_capital_after_cost = today_capital_before_cost - today_cost
#
# 			daily_capital.append(today_capital_after_cost)
# 		else:
# 			daily_capital.append(today_capital_before_cost)
#
#
# 	else: # 以后的每一天
#
# 		# 今天的持股数量 = 今天开盘前总资产*今天开盘后要有的总持股市值占比／今天开盘价
# 		day_two_shares =  daily_capital[idx-1]*index_preds_target[idx,0]/open_prices[idx]
#
# 		# 昨日与今日的持股数之差
# 		daily_share_difference = np.abs(day_two_shares - daily_stock_shares[idx-1])
#
# 		# 第二天及以后的变化 + 阀值控制 + 交易成本计算
# 		# 阀值： 如果， 第二天的交易股数占比 < 0.85; 那么，不做交易，持仓不动；也就是今天的持股数，要跟昨天一样；今天的总资产的方法，有变化
# 		if threshold and daily_share_difference < 0.88:
# 			# 忽略噪音，不做交易
# 			daily_share_difference = 0.0
# 			# 今天的持股数，与昨天保持一致; 姑且认为昨天和今天的持仓市值占比也一样
# 			daily_stock_shares.append(daily_stock_shares[idx-1])
# 			index_preds_target[idx, 0] = index_preds_target[idx-1, 0]
# 			# 今天的总资产 = 昨天总资产 + 今天持仓市值*今天价格变化率 = 昨天总资产 + 今天持股数*今天开盘价*今天价格变化率
# 			today_capital_threshold = daily_capital[idx-1] + daily_stock_shares[idx]*open_prices[idx]*index_preds_target[idx,1]
# 			today_capital_before_cost = today_capital_threshold
# 		else: # 如果没有阀值，或者在阀值之外，今天持股数不能等同于昨天持股数，必须根据今天持仓市值／开盘价来计算
# 			daily_stock_shares.append(day_two_shares)
#
# 			# 如果不计算交易成本
# 			# 今天的总资产=昨天总资产*（1+今天开盘后要买入的市值占比*今日收盘与昨日收盘价格变化率）
# 			# 如果需要也可以使用， 今天的总资产=昨天总资产*（1+今天开盘后要买入的市值占比*今日收盘与昨日收盘价格变化率），只需要修改目标值生成的计算方法就行。
# 			today_capital_before_cost = daily_capital[idx-1]*(1 + index_preds_target[idx,0]*index_preds_target[idx,1])
#
#
#
#
# 		# 第二天的市值（计算交易成本）
# 		# 今天的交易成本 = 昨天和今天持股数之差 * 今天开盘价 * 0.001
# 		today_cost = daily_share_difference * open_prices[idx] * 0.001
#
# 		# 今天计算交易成本后的总市值 = 今天的总市值（不计算成本） - 今天交易成本
# 		today_capital_after_cost = today_capital_before_cost - today_cost
#
#
# 		# 如果需要计算交易成本
# 		if add_cost:
# 			daily_capital.append(today_capital_after_cost)
# 		else:
# 			daily_capital.append(today_capital_before_cost)
#
#
# ### 统计总收益
# accum_profit = np.array(daily_capital)-1 # 累积总资产减去初始资产 = 累积收益
# print("final date:", date[-1])
# print("final_return:", accum_profit[-1])
#
# ##### accumulation of transaction percentage，即换手率
# # 这里我们用市值占比率，从0到1， 从1到0， 来理解计算换手率
# preds = index_preds_target[:,0] # 每日持仓的总资产占比
# changes_preds = np.abs(preds[1:] - preds[:-1]) # 相邻两日的持仓占比之差（变化）
# turnover_rate = np.cumsum(changes_preds) # 累积差值，获得总资产进出市场的次数

################################################################
# price_curve_prediction_continuous_color_simple
# plot close price curve and fill prediction array as price curve color
################################################################
color_data = index_preds_target[:,0]
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
ax1 = plt.subplot2grid((6, 3), (0, 0), colspan=3, rowspan=4)
#############
### plot close_price curve and fill predictions as continuous color ######
#############
for start, stop, col in zip(xy[:-1], xy[1:], color_data):
    x, y = zip(start, stop)
    ax1.plot(x, y, color=uniqueish_color(col))
ax1.plot(accum_profit, c='gray', alpha=0.5, label='accum_profit')
ax1.legend(loc='best')
ax1.set_title('sigmoid_model, ETF50, No_cost**, from %s to %s return: %04f' % (date[0], date[-1], accum_profit[-1]))

#############
### plot 换手率
#############
ax2 = plt.subplot2grid((6, 3), (4, 0), colspan=3, rowspan=2)
ax2.plot(turnover_rate, c='red', label='turnover_rate')
ax2.legend(loc='best')
ax2.set_title("ETF50 TurnOver Rate: %02f" % turnover_rate[-1])

### plot pos as bars or as color map
# ax3 = plt.subplot2grid((10, 3), (6, 0), colspan=3, rowspan=4)
# X = np.arange(len(index_preds_target))
# ax3.bar(X, index_preds_target[:,0], facecolor='#9999ff', edgecolor='blue')
# ax3.set_title('pos as bars') # change model name

plt.tight_layout()
plt.show()


# img = index_preds_target[:,0].reshape((-1,1)).transpose((1,0))
# imgs = np.concatenate((img, img, img, img, img, img),axis=0)
# plt.imshow(imgs, cmap='binary')
# plt.title("pos as color array")
# plt.show()


# plt.savefig("/Users/Natsume/Downloads/data_for_all/stocks/model_performance/ETF300_model4_addcost.png")
