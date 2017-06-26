# source from https://github.com/happynoom/DeepTrade_keras

"""
### from OHLCV arrays to indicator_features array, price_change_target array

Inputs:
1. import previous code file, got OHLCV arrays on a csv file;
2. a number of user selected indicators, which are supported internally by self.extract_by_type();
3. set window in extract_feature(), e.g. 30 days;

Return:
1. features arrays,
2. targets array


- given OHLCV arrays imported as global variables
- 30 days as window, user_selected_indicators
- use indicators.moving_extract() to get indicator_features array, target array
	- user select or define 61 indicators
	- 5 arrays OHLCV become single array (61, 6464)
	- use 30 days window to split this array above, we get a list of 6434 arrays of shape (61, 30)
	- convert list to array shape (6434, 61, 30), now we got features array
	- and flatten 61*30 for saving convenience (or not)
	- use closes array, with 30 days window, start from 30th day, we get daily price change pct as targets array (6434, )
	- finally, return (moving_features_array, moving_targets_array)
"""

import talib
import math
import numpy as np



class IndicatorCreator(object):
    """
    1. self.selector: user selected indicators (only applicable if matched with internally supported indicators)
    2. self.supported: show internal supported indicators
    3. self.feature: empty list to store converted indicator data later
    """

    def __init__(self, selector):
        self.selector = selector
        self.supported = {"ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"}
        self.feature = []

    """
    1. convert a list of 61 arrays (6464, ) to a single array (61, 6464)
    2. convert (61, 6464) to a list of 6434 arrays (61, 30)
    3. convert the above list to array (6434, 61*30)
    4. create targe array (6434, ) of price change start from 30th day onward
    5. return feature array (6434, 61*30), target array (6434, )
    """
    def moving_extract(self, window=30, open_prices=None, close_prices=None, high_prices=None, low_prices=None,
                       volumes=None, with_label=True, flatten=True):
        self.extract(open_prices=open_prices, close_prices=close_prices, high_prices=high_prices, low_prices=low_prices,
                     volumes=volumes) # self.feature is a list with 61 indicators
        feature_arr = np.asarray(self.feature) # list to array, shape (61, 6464) = (num_indicators, num_days)
        p = 0
        rows = feature_arr.shape[0]
        print("feature dimension: %s" % rows)
        if with_label:
            moving_features = []
            moving_labels = []
            while p + window < feature_arr.shape[1]: # start from 30th day, loop for each everyday onward
                x = feature_arr[:, p:p + window] # take all indicators for 30 days range, start from day 1, then start from day 2, and so on
                # p_change: (the day-30th-price - day_29th_price)/day_29th_price
                p_change = (close_prices[p + window] - close_prices[p + window - 1]) / close_prices[p + window - 1]
                # use percent of change as label
                y = p_change
                if flatten:
                    x = x.flatten("F") # from (61, 30) to (61*30, )
                moving_features.append(np.nan_to_num(x)) # 1. convert nan to 0.0; 2. store x a 30_days_61_indicators as a long vector into list moving_features
                moving_labels.append(y)
                p += 1
			# now moving_features is a list of 6434 == 6464 - 30
            return np.asarray(moving_features), np.asarray(moving_labels)
        else:
            moving_features = []
            while p + window <= feature_arr.shape[1]:
                x = feature_arr[:, p:p + window]
                if flatten:
                    x = x.flatten("F")
                moving_features.append(np.nan_to_num(x))
                p += 1
            return moving_features


    """
    1. check every user provided indicators, see whether match with internally supported indicators
    2. if so, self.extract_by_type() will create each indicator array from arrays of OHLCV
    3. save all indicator arrays into ChartFeature.feature (list)
    """

    def extract(self, open_prices=None, close_prices=None, high_prices=None, low_prices=None, volumes=None):
        self.feature = []
        for feature_type in self.selector:
            if feature_type in self.supported:
                print("extracting feature : %s" % feature_type)
                self.extract_by_type(feature_type, open_prices=open_prices, close_prices=close_prices,
                                     high_prices=high_prices, low_prices=low_prices, volumes=volumes)
            else:
                print("feature type not supported: %s" % feature_type)
        return self.feature

    """
    1. given an user selected indicator name, and arrays of OHLCV
    2. if the indicator match with internal supported indicators, make the indicator array (or arrays)
    3. then append the indicator array(s) into self.feature (list)
    """
    def extract_by_type(self, feature_type, open_prices=None, close_prices=None, high_prices=None, low_prices=None,
                        volumes=None):
        if feature_type == 'ROCP': # price change pct daily basis on close
            rocp = talib.ROCP(close_prices, timeperiod=1)
            self.feature.append(rocp)
        if feature_type == 'OROCP': # price change pct daily basis on open
            orocp = talib.ROCP(open_prices, timeperiod=1)
            self.feature.append(orocp)
        if feature_type == 'HROCP':
            hrocp = talib.ROCP(high_prices, timeperiod=1)
            self.feature.append(hrocp)
        if feature_type == 'LROCP':
            lrocp = talib.ROCP(low_prices, timeperiod=1)
            self.feature.append(lrocp)
        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            norm_macd = np.nan_to_num(macd) / math.sqrt(np.var(np.nan_to_num(macd)))
            norm_signal = np.nan_to_num(signal) / math.sqrt(np.var(np.nan_to_num(signal)))
            norm_hist = np.nan_to_num(hist) / math.sqrt(np.var(np.nan_to_num(hist)))
            macdrocp = talib.ROCP(norm_macd + np.max(norm_macd) - np.min(norm_macd), timeperiod=1)
            signalrocp = talib.ROCP(norm_signal + np.max(norm_signal) - np.min(norm_signal), timeperiod=1)
            histrocp = talib.ROCP(norm_hist + np.max(norm_hist) - np.min(norm_hist), timeperiod=1)
            # self.feature.append(macd / 100.0)
            # self.feature.append(signal / 100.0)
            # self.feature.append(hist / 100.0)
            self.feature.append(norm_macd)
            self.feature.append(norm_signal)
            self.feature.append(norm_hist)

            self.feature.append(macdrocp)
            self.feature.append(signalrocp)
            self.feature.append(histrocp)
        if feature_type == 'RSI':
            rsi6 = talib.RSI(close_prices, timeperiod=6)
            rsi12 = talib.RSI(close_prices, timeperiod=12)
            rsi24 = talib.RSI(close_prices, timeperiod=24)
            rsi6rocp = talib.ROCP(rsi6 + 100., timeperiod=1)
            rsi12rocp = talib.ROCP(rsi12 + 100., timeperiod=1)
            rsi24rocp = talib.ROCP(rsi24 + 100., timeperiod=1)
            self.feature.append(rsi6 / 100.0 - 0.5)
            self.feature.append(rsi12 / 100.0 - 0.5)
            self.feature.append(rsi24 / 100.0 - 0.5)
            # self.feature.append(np.maximum(rsi6 / 100.0 - 0.8, 0))
            # self.feature.append(np.maximum(rsi12 / 100.0 - 0.8, 0))
            # self.feature.append(np.maximum(rsi24 / 100.0 - 0.8, 0))
            # self.feature.append(np.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(np.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(np.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(np.maximum(np.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(np.maximum(np.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(np.maximum(np.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            self.feature.append(rsi6rocp)
            self.feature.append(rsi12rocp)
            self.feature.append(rsi24rocp)
        if feature_type == 'VROCP':
            # vrocp = talib.ROCP(volumes, timeperiod=1)
            norm_volumes = (volumes - np.mean(volumes)) / math.sqrt(np.var(volumes))
            vrocp = talib.ROCP(norm_volumes + np.max(norm_volumes) - np.min(norm_volumes), timeperiod=1)
            # self.feature.append(norm_volumes)
            self.feature.append(vrocp)
        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            self.feature.append((upperband - close_prices) / close_prices)
            self.feature.append((middleband - close_prices) / close_prices)
            self.feature.append((lowerband - close_prices) / close_prices)
        if feature_type == 'MA':
            ma5 = talib.MA(close_prices, timeperiod=5)
            ma10 = talib.MA(close_prices, timeperiod=10)
            ma20 = talib.MA(close_prices, timeperiod=20)
            ma30 = talib.MA(close_prices, timeperiod=30)
            ma60 = talib.MA(close_prices, timeperiod=60)
            ma90 = talib.MA(close_prices, timeperiod=90)
            ma120 = talib.MA(close_prices, timeperiod=120)
            ma180 = talib.MA(close_prices, timeperiod=180)
            ma360 = talib.MA(close_prices, timeperiod=360)
            ma720 = talib.MA(close_prices, timeperiod=720)
            ma5rocp = talib.ROCP(ma5, timeperiod=1)
            ma10rocp = talib.ROCP(ma10, timeperiod=1)
            ma20rocp = talib.ROCP(ma20, timeperiod=1)
            ma30rocp = talib.ROCP(ma30, timeperiod=1)
            ma60rocp = talib.ROCP(ma60, timeperiod=1)
            ma90rocp = talib.ROCP(ma90, timeperiod=1)
            ma120rocp = talib.ROCP(ma120, timeperiod=1)
            ma180rocp = talib.ROCP(ma180, timeperiod=1)
            ma360rocp = talib.ROCP(ma360, timeperiod=1)
            ma720rocp = talib.ROCP(ma720, timeperiod=1)
            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            self.feature.append(ma360rocp)
            self.feature.append(ma720rocp)
            self.feature.append((ma5 - close_prices) / close_prices)
            self.feature.append((ma10 - close_prices) / close_prices)
            self.feature.append((ma20 - close_prices) / close_prices)
            self.feature.append((ma30 - close_prices) / close_prices)
            self.feature.append((ma60 - close_prices) / close_prices)
            self.feature.append((ma90 - close_prices) / close_prices)
            self.feature.append((ma120 - close_prices) / close_prices)
            self.feature.append((ma180 - close_prices) / close_prices)
            self.feature.append((ma360 - close_prices) / close_prices)
            self.feature.append((ma720 - close_prices) / close_prices)
        if feature_type == 'VMA':
            ma5 = talib.MA(volumes, timeperiod=5)
            ma10 = talib.MA(volumes, timeperiod=10)
            ma20 = talib.MA(volumes, timeperiod=20)
            ma30 = talib.MA(volumes, timeperiod=30)
            ma60 = talib.MA(volumes, timeperiod=60)
            ma90 = talib.MA(volumes, timeperiod=90)
            ma120 = talib.MA(volumes, timeperiod=120)
            ma180 = talib.MA(volumes, timeperiod=180)
            ma360 = talib.MA(volumes, timeperiod=360)
            ma720 = talib.MA(volumes, timeperiod=720)
            ma5rocp = talib.ROCP(ma5, timeperiod=1)
            ma10rocp = talib.ROCP(ma10, timeperiod=1)
            ma20rocp = talib.ROCP(ma20, timeperiod=1)
            ma30rocp = talib.ROCP(ma30, timeperiod=1)
            ma60rocp = talib.ROCP(ma60, timeperiod=1)
            ma90rocp = talib.ROCP(ma90, timeperiod=1)
            ma120rocp = talib.ROCP(ma120, timeperiod=1)
            ma180rocp = talib.ROCP(ma180, timeperiod=1)
            ma360rocp = talib.ROCP(ma360, timeperiod=1)
            ma720rocp = talib.ROCP(ma720, timeperiod=1)
            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            self.feature.append(ma360rocp)
            self.feature.append(ma720rocp)
            self.feature.append((ma5 - volumes) / (volumes + 1))
            self.feature.append((ma10 - volumes) / (volumes + 1))
            self.feature.append((ma20 - volumes) / (volumes + 1))
            self.feature.append((ma30 - volumes) / (volumes + 1))
            self.feature.append((ma60 - volumes) / (volumes + 1))
            self.feature.append((ma90 - volumes) / (volumes + 1))
            self.feature.append((ma120 - volumes) / (volumes + 1))
            self.feature.append((ma180 - volumes) / (volumes + 1))
            self.feature.append((ma360 - volumes) / (volumes + 1))
            self.feature.append((ma720 - volumes) / (volumes + 1))
        if feature_type == 'PRICE_VOLUME':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            norm_volumes = (volumes - np.mean(volumes)) / math.sqrt(np.var(volumes))
            vrocp = talib.ROCP(norm_volumes + np.max(norm_volumes) - np.min(norm_volumes), timeperiod=1)
            # vrocp = talib.ROCP(volumes, timeperiod=1)
            pv = rocp * vrocp * 100.
            self.feature.append(pv)


"""
- given OHLCV arrays imported as global variables
- 30 days as window, user_selected_indicators
- use indicators.moving_extract() to get indicator_features array, target array
	- user select or define 61 indicators
	- 5 arrays OHLCV become single array (61, 6464)
	- use 30 days window to split this array above, we get a list of 6434 arrays of shape (61, 30)
	- convert list to array shape (6434, 61, 30), now we got features array
	- and flatten 61*30 for saving convenience
	- use closes array, with 30 days window, start from 30th day, we get daily price change pct as targets array (6434, )
	- finally, return (moving_features_array, moving_targets_array)

"""
def extract_feature(selector, window=30, with_label=True, flatten=False):
	# use user_selected_indicators to create IndicatorCreator object
    indicators = IndicatorCreator(selector)

    if with_label:
        moving_features, moving_labels = indicators.moving_extract(window=window, open_prices=opens, close_prices=closes, high_prices=highs, low_prices=lows, volumes=volumes, with_label=with_label, flatten=flatten)
		# extract_feature: func outside ChartFeature class
		# dive into ChartFeature.moving_extract():
		# then into ChartFeature.extract(): extract one feature group at a time
		# ChartFeature.feature is a list has 61 indicators in total
        return moving_features, moving_labels
    else:
        moving_features = indicators.moving_extract(window=window, open_prices=opens, close_prices=closes,
                                                       high_prices=highs, low_prices=lows, volumes=volumes,
                                                       with_label=with_label, flatten=flatten)
        return moving_features

# import OHLCV arrays globally
from prep_data_03_stock_01_csv_2_objects_2_arrays_DOHLCV import opens, highs, lows, closes, volumes

# all internal supported indicators are selected here
user_indicators = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"]

# get features array and target array
moving_indicators_features, moving_real_price_changes = extract_feature(selector=user_indicators)

moving_real_price_changes.shape
