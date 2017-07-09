"""
Uses: func: get_stock_preds_target
1. load the best trained model (Note: BatchRenormalization can only be loaded using wp object)
1. convert a stock csv into an preds_target array

Inputs:
1. stock csv file_path

Return:
1. array: preds_target

"""



from tensorflow.contrib.keras.python.keras.models import load_model, Model
from tensorflow.contrib.keras.python.keras import backend as K
from build_model_03_stock_02_build_compile_model_with_WindPuller_init import wp
from prep_data_03_stock_05_get_stock_features_targets_from_csv import stock_csv_features_targets
import numpy as np
# from renormalization import BatchRenormalization

def get_stock_preds_target(stock_path):
# get mdjt data ready
	global wp

	stock_features, stock_target = stock_csv_features_targets(stock_path)


	# import BatchRenormalization or global BatchRenormalization won't help load_model()
	# only wp.load_model() can bring BatchRenormalization into globals()
	# wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/best_models_trained/model4_new.best") # xiaoyu trained this new model
	# wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/floyd_model2_1000/floyd_model2_1000.h5") # floyd model2 1000 epochs, but lr = 0.002
	# wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/best_models_trained/model.30.4_bluechips_indices.best")
	# wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/best_models_trained/model.30.best")
	wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/best_models_trained/model.30.best.bias_removed")
	# error on unknown layer BatchRenormalization, if use the following line
	# wp = wp.load_model("/Users/Natsume/Downloads/DeepTrade_keras/author_log_weights/model.30.best")
	# wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/qcg_train_only_ma/qcg_model_2000.h5")

	# predict with model on training, validation and test sets
	pred_capital_pos = wp_best.predict(stock_features)
	pred_capital_pos = np.reshape(pred_capital_pos, (-1, 1))
	true_price_change = np.reshape(stock_target, (-1, 1))

	preds_target = np.concatenate((pred_capital_pos, true_price_change), axis=1)

	return preds_target

"""
Example
"""
# mdjt_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices/mdjt_prices.csv"
# #
# mdjt_preds_target = get_stock_preds_target(mdjt_path)
