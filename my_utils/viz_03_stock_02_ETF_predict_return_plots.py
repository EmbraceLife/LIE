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
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index000001.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF50.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF500.csv"
index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF300.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index50.csv"

date,open_prices,_,_, closes, _ = csv_df_arrays(index_path)


from predict_model_03_stock_01_predict_stock_best_model_save_result import get_stock_preds_target
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
# time_span = 700  # 从今天回溯700 days
# time_span = 90  # 从今天回溯90 days
# time_span = 30  # 从今天回溯30 days
# time_span = 1  # 从今天回溯1 days
time_span = 4  # 昨天开始交易，到今天收盘，交易开始两天了


# zoom in and out for the last 700 trading days
open_prices = open_prices[-time_span:]
closes = closes[-time_span:]
index_preds_target = index_preds_target[-time_span:]



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
### get return curve dataset
# target_closes = closes/closes[0]# normalized price to price changes
# open_prices = open_prices/open_prices[0]# normalized price to price changes
# daily_capital=[]
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
### get return curve dataset
# target_closes = closes/closes[0]# normalized price to price changes
# open_prices = open_prices/open_prices[0]# normalized price to price changes
# daily_capital=[]

# # 原理： 从最简单的变化入手，现有模型上修改，现有交易理解上改进
# # 改进预测值理解： 不是市值占比，而是持股数占比
# # 持股总数不变，可以提前设置，比如总数设置为10000股； 预测总股数比率， 80% 就是8000股；
# # 设置：总投资额 = 1； 总持股数 = 1
# # 预测值 or index_preds_target[0,0] == 预测当日开盘后需要的持股数占比
# init_capital = 1.0
# daily_capital = []
# daily_trade_pct = []
# add_cost = True
# threshold = True
#
# # 第一天的交易股数占比：当前预测值， index_preds_target[0,0]
# # 第一天的市值： 第一天开盘买入，当天结束时的市值 = 当天总资产（比如，1） + 当天持仓市值在收盘时的增值部分 = 当天总资产  + 当天开盘后形成的持股数占比 * 当天开盘价 * 当天价格变化比 = 1 + index_preds_target[0,0] * open_prices[0] * index_preds_target[0,1]
# day_one_capital = 1 + index_preds_target[0,0] * open_prices[0] * index_preds_target[0,1]
# # 第一天的交易成本： 当天开盘后形成的持股数占比 * 当天开盘价 * 0.001 = index_preds_target[0,0] * open_prices[0] * 0.001
# day_one_trade_pct = index_preds_target[0,0]
# day_one_cost = day_one_trade_pct * open_prices[0] * 0.001
# # 第一天收盘实际市值： 第一天的市值 - 第一天的交易成本
# day_one_capital_add_cost = day_one_capital - day_one_cost
# # 收集每日交易股数占比
# daily_trade_pct.append(day_one_trade_pct)
# # 收集
# if add_cost:
# 	daily_capital.append(day_one_capital_add_cost)
#
# else:
# 	daily_capital.append(day_one_capital)
#
#
# # 第二天及以后的变化：
# for idx in range(1, len(index_preds_target)):
#
# # 第二天的交易股数占比： 当前预测值 - 昨天预测值 = index_preds_target[idx,0] - index_preds_target[idx-1,0]
# 	day_two_trade_pct = np.abs(index_preds_target[idx,0] - index_preds_target[idx-1,0])
#
# ############################
# # threshold_reduce_noise_trade
# # 第二天及以后的变化 + 阀值控制
# # 第二天的交易股数占比： 当前预测值 - 昨天预测值
# # 阀值： 如果， 第二天的交易股数占比 < 0.1; 那么，当前预测值 = 昨天预测值; 即，不做交易，继续持有不变
# 	if threshold and day_two_trade_pct < 0.85:
# 		day_two_trade_pct = 0.0
# 		index_preds_target[idx,0] = index_preds_target[idx-1,0]
#
# # 第二天的市值： 第二天开盘买入，当天结束时的市值 = 昨天总资产 + 当天持仓市值在收盘时的增值部分 = 昨天收盘总资产  + 当天开盘后形成的持股数占比 * 当天开盘价 * 当天价格变化 = 昨天收盘总资产 + index_preds_target[idx,0] * open_prices[idx] * index_preds_target[idx,1]
# 	day_two_capital = daily_capital[idx-1] + index_preds_target[idx,0] * open_prices[idx] * index_preds_target[idx,1]
#
#
# # 第二天的交易成本： 第二天的交易股数占比 * 0.001 * 第二天的开盘价
# 	day_two_cost = day_two_trade_pct * open_prices[idx] * 0.001
# # 第二天收盘实际市值： 第二天的市值 - 第二天的交易成本
# 	day_two_capital_add_cost = day_two_capital - day_two_cost
#
# # 收集每日交易股数占比
# 	daily_trade_pct.append(day_two_trade_pct)
#
# # 收集
# 	if add_cost:
# 		daily_capital.append(day_two_capital_add_cost)
#
# 	else:
# 		daily_capital.append(day_two_capital)
#
# ### 统计总收益
# accum_profit = np.array(daily_capital)-1 # 累积总资产减去初始资产 = 累积收益
# print("final date:", date[-1])
# print("final_return:", accum_profit[-1])
#
# # 如何计算换手率？
# # 所以第二天交易股数占比总和
# turnover_rate = np.cumsum(np.array(daily_trade_pct))

######################################################
# revised trading cost based on MonteCarlo's suggestion
#####################################################
### get return curve dataset
# target_closes = closes/closes[0]# normalized price to price changes
# open_prices = open_prices/open_prices[0]# normalized price to price changes
# daily_capital=[]

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

# # color_data 经过阀值调节过的预测值
# color_data = index_preds_target[:,0]





##############################################################################
# MonteCarlo: 尽可能还原真实交易场景，不考虑无交易成本的情况
# 	- 应该是openprice
# 	- day1，有现金（初始市值）A元，预测市值占比为a1，那么股数为A*a1/openprice，付出手续费=股数*openprice*0.001=A*a1*0.001，剩余现金=A*（1-a1)-手续费
# 	- day2，日初市值=前一日股数*前一日closeprice+前一日剩余现金；今日股数=日初市值*今日预测市值占比a2/openprice，手续费=abs(今日股数-昨日股数)*openprice**0.001，剩余现金=日初市值（1-a2）-今日手续费
##############################################################################
init_capital = 1000000 # 初始总现金
y_pred = index_preds_target[:,0] # 预测值，当日持仓市值占比
y_true = index_preds_target[:,1] # 相邻两天收盘价变化率
open_prices = open_prices # 每日开盘价，标准化处理 (受否必须再减1？)
closes = closes # 每日收盘价，标准化处理
daily_shares_pos = [] # 用于收集每日的持股数，让实际交易便捷
daily_cash_left = [] # 用于收集每日的现金剩余量
daily_capital = [] # 用于收集每日收盘时总资金
daily_differences = [] # 用于收集每日持股变化或买卖情况，有正有负
daily_action_on = [] # 用于收集每日是否交易， [true or false]
daily_costs = [] # 用于收集每日交易成本
shares_pos_no0 = [] # 用于收集非0每日持股数量
# 第一天的值
daily_shares_pos.append(0) # 用于收集每日的持股数，让实际交易便捷
daily_cash_left.append(init_capital) # 用于收集每日的现金剩余量
daily_capital.append(init_capital) # 用于收集每日收盘时总资金
daily_differences.append(0) # 用于收集每日持股变化或买卖情况，有正有负
daily_action_on.append(False) # 用于收集每日是否交易， [true or false]
daily_costs.append(0) # 用于收集每日交易成本

use_threshold = True
threshold = 0.9

for idx in range(len(y_pred)-1):
	if idx == 0: # 第二天
		y_pred[idx] # 第二天开盘的预测市值占比
		open_prices[idx+1] # 第一日开盘价
		# y_pred[idx] can be 0 or non 0 here
		shares_pos = daily_capital[idx] * y_pred[idx] / open_prices[idx+1]  # 第一天的持仓股数, 四舍五入，取100整数值
		daily_shares_pos.append(shares_pos) # 收集第一天的持股数

		# record all shares_pos are non 0
		if shares_pos != 0.0:
			shares_pos_no0.append(shares_pos)

		daily_differences.append(shares_pos) # 收集第一天的持股差
		# 如果第二天的交易差不是0，那么交易；否则不交易
		if daily_differences[idx+1] != 0.0:
			daily_action_on.append(True)
		else:
			daily_action_on.append(False)

		# 因为持仓股数可以是0， 所以交易成本也可以是0
		cost = shares_pos * open_prices[idx+1] * 0.001 # = init_capital * y_pred[idx] * 0.001 = 持股数*股价*0.001 = 持仓市值*0.001 = 交易成本
		# 第一天只要预测值不是0，交易成本就免不了；不用考虑阀值
		daily_costs.append(cost)

		cash_left = daily_capital[idx] * (1 - y_pred[idx]) - cost # 剩余现金 = 市值意外的现金 - 交易成本
		daily_cash_left.append(cash_left)
		capital_day_2 = daily_shares_pos[idx+1] * closes[idx+1] + daily_cash_left[idx+1] # 当日收盘时的总资金 = 当日持仓股数*当日收盘价 + 当日剩余现金
		daily_capital.append(capital_day_2) # 收集第一日总资金




	else: # 从第三天起
		closes[idx] # 前一日的收盘价
		daily_shares_pos[idx] # 前一日的持股数
		daily_cash_left[idx] # 前一日现金剩余
		daily_capital[idx] # 当日（即第二日）开盘前的总资金, 即第一日收盘时的总资金

		y_pred[idx] # 当日预测市值占比
		open_prices[idx+1] # 当日开盘价

		# 第三天的预测值可以是0
		# shares_pos = daily_capital[idx] * y_pred[idx] / open_prices[idx+1] # use next day's open price is more accurate, but not convenient to prepare
		shares_pos = daily_capital[idx] * y_pred[idx] / closes[idx]
		daily_shares_pos.append(shares_pos) # 收集第二天的持股数

		# 通过阀值来降低噪音和交易频率
		daily_differences.append(daily_shares_pos[idx+1] - daily_shares_pos[idx]) # 收集第三天的持股差，可负可正
		daily_share_difference = np.abs(daily_shares_pos[idx+1] - daily_shares_pos[idx]) # 昨天和今天的持股差值的绝对值

		# 如果昨天持股数不是0
		if daily_shares_pos[idx] != 0.0:
			# 昨天与今天持仓股数差值与昨天持股数的比率，如下
			daily_share_difference_rate = daily_share_difference/daily_shares_pos[idx]
		else: # 如果昨天持股数是0，那么差值比也是0
			if len(shares_pos_no0) != 0:
				daily_share_difference_rate=daily_share_difference/shares_pos_no0[-1]
			else:
				daily_share_difference_rate = 0.0



		# 如果持股差值在阀值范围之内，那么不交易
		if use_threshold and daily_share_difference_rate < threshold:
			daily_share_difference_rate = 0.0 # 将持股差值比化为0
			daily_share_difference = 0.0 # 让持股差值化为0
			daily_shares_pos[idx+1] = daily_shares_pos[idx] # 将今日持股数维持昨日持股数

			daily_cash_left.append(daily_cash_left[idx])
			daily_action_on.append(False) # 如果股数变化在降噪范围之内，不做交易
		else: # 如果在阀值以外或者忽略阀值，
			daily_action_on.append(True)
			cash_left = daily_capital[idx] * (1 - y_pred[idx]) - cost
			daily_cash_left.append(cash_left) # 当日现金结余 = 今日开盘前（即昨日收盘时）总资金 * （1-今日市值在总资产占比）- 今日交易成本

		cost = daily_share_difference * open_prices[idx+1] * 0.001 # = 今日交易成本 = |今日与昨日持仓股数之差| * 当日开盘价 * 0.001
		daily_costs.append(cost)

		# 收集当日持股数量，非0
		shares_pos_no0.append(daily_shares_pos[idx+1])
		# 当日收盘时的总资金 = 当日持股数 * 当日收盘价 + 当日现金结余 = 当日收盘市值 + 当日现金结余
		capital_day_2 = daily_shares_pos[idx+1] * closes[idx+1] + daily_cash_left[idx+1]
		daily_capital.append(capital_day_2) # 收集当日总资金


# 计算累积总资金曲线
accum_profit = (np.array(daily_capital)/daily_capital[0])-1# 累积总资金曲线， 减去初始资金 1，获得收益累积曲线
print("final date:", date[-1]) # 最近日期
print("final_return:", accum_profit[-1]) # 累积总收益
###############
print("Now, after today's trading, before tomorrow morning ....")
print("today's open price:", open_prices[-1]) # 当日收盘价
print("today's close price:", closes[-1]) # 当日收盘价
print("prediction for yesterday morning action:", y_pred[-3])
print("prediction for morning action:", y_pred[-2])
print("prediction for tomorrow: ", y_pred[-1]) # 预测明早的市值占比


estimate_yesterdayMorning_shares_hold = daily_shares_pos[-2]
print("estimate how many shares to hold yesterday morning:", estimate_yesterdayMorning_shares_hold) # 预测明早持股数量
estimate_thisMorning_shares_hold = daily_shares_pos[-1]
print("estimate how many shares to hold this morning:", estimate_thisMorning_shares_hold) # 预测明早持股数量
### how many shares to hold tomorrow morning
estimate_tomorrow_shares_hold = np.round(daily_capital[-1] * y_pred[-1] / closes[-1], -2)
print("estimate how many shares to hold tomorrow:", estimate_tomorrow_shares_hold) # 预测明早持股数量

#### how much capital do we have tomorrow morning
print("end of day_before_yesterday's capital: ", daily_capital[-3]) # 当日收盘时总资产
print("end of yesterday's capital: ", daily_capital[-2]) # 当日收盘时总资产
print("end of today's capital: ", daily_capital[-1]) # 当日收盘时总资产

#### how much trade cost and cash left today
print("today's trading cost: ", daily_costs[-1])
print("today's cash left:", daily_cash_left[-1])
### check out all y_preds and daily_shares_pos and daily_capital so final_return
print("all predictions so far: ", y_pred)
print("all positions so far:", daily_shares_pos)
print("all capitals so far:", daily_capital)


# 换手率曲线
# 换手率：定义理解，持股数从1到0，从0到1， 一共产生2次换手；但不知道要除去什么来获得比率？所以，尝试计算换手的股票总数 （下面）
# 换手的股票总数：相邻两天持仓股数差值绝对值的总和
# turnover1 = np.array([0] + daily_shares_pos[:-1]) # [:, -1) to note!!
# turnover2 = np.array(daily_shares_pos)
# daily_share_diffs = turnover2 - turnover1
# turnover_rate = np.cumsum(daily_share_diffs) # 换手的累积股数值

# ##### accumulation of transaction percentage，即换手率
# # 这里我们用市值占比率，从0到1， 从1到0， 来理解计算换手率
# y_pred # 通过阀值更新后实际每日持仓的总资产占比
# changes_preds = np.abs(y_pred[1:] - y_pred[:-1]) # 相邻两日的持仓占比之差（变化）
# turnover_rate = np.cumsum(changes_preds) # 累积差值，获得总资产进出市场的次数

daily_shares_pos # 通过阀值更新后实际每日持仓的总资产占比
changes_pos_rate = np.abs(np.array(daily_shares_pos[1:]) - np.array(daily_shares_pos[:-1]))/np.array(daily_shares_pos).max() # 相邻两日的持仓占比之差（变化）
turnover_rate = np.cumsum(changes_pos_rate) # 累积差值，获得总资产进出市场的次数

## color data for close prices： 经过阀值调节过的预测值
color_data = y_pred
## 用于画收盘价曲线的数据
target_closes = closes/closes[0]

################################################################
# price_curve_prediction_continuous_color_simple
# plot close price curve and fill prediction array as price curve color
################################################################
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
ax1 = plt.subplot2grid((8, 3), (0, 0), colspan=3, rowspan=4)
#############
### plot close_price curve and fill predictions as continuous color ######
#############
for start, stop, col in zip(xy[:-1], xy[1:], color_data):
    x, y = zip(start, stop)
    ax1.plot(x, y, color=uniqueish_color(col))

ax1.plot(accum_profit, c='gray', alpha=0.5, label='accum_profit')
ax1.legend(loc='best')
ax1.set_title('sigmoid_mock_90_etf300 from %s to %s return: %04f' % (date[0], date[-1], accum_profit[-1]))

#############
### plot 换手率
#############
ax2 = plt.subplot2grid((8, 3), (4, 0), colspan=3, rowspan=2)
ax2.plot(turnover_rate, c='red', label='turnover_rate')
ax2.legend(loc='best')
ax2.set_title("TurnOver Rate: %02f" % turnover_rate[-1])

### plot daily_shares_pos curve
ax3 = plt.subplot2grid((8, 3), (6, 0), colspan=3, rowspan=2)
init_shares_full = init_capital/open_prices[0]
daily_shares_rate = np.array(daily_shares_pos)/init_shares_full
ax3.plot(daily_shares_rate, c='k', label='daily_shares_rate')
ax3.legend(loc='best')
ax3.set_title('init_share_number: %d, latest_share_number: %d' % (init_shares_full, daily_shares_pos[-1])) # change model name

plt.tight_layout()
plt.show()


# img = index_preds_target[:,0].reshape((-1,1)).transpose((1,0))
# imgs = np.concatenate((img, img, img, img, img, img),axis=0)
# plt.imshow(imgs, cmap='binary')
# plt.title("pos as color array")
# plt.show()


# plt.savefig("/Users/Natsume/Downloads/data_for_all/stocks/model_performance/ETF300_model4_addcost.png")
