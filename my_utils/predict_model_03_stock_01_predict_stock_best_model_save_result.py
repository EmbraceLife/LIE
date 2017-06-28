"""
Inputs:
1. import an empty but compiled wp
2. best_model_path
3. paths for storing train_pos_targets, valid_pos_targets, test_pos_targets

Returns:
1. predictions with the best model on training, valid and test set (features, targets arrays)
2. save predictions and targets into a single array and store them in files

### Summary
- initiate an empty model from WindPuller
- load the best trained model into it
- use this model to predict training, validation, testing sets
- save predictions and targets into a single array, and save them into files
"""



from tensorflow.contrib.keras.python.keras.models import load_model
from build_model_03_stock_02_build_compile_model_with_WindPuller_init import wp
from prep_data_03_stock_05_get_stock_features_targets_from_csv import stock_csv_features_targets
import numpy as np

def get_stock_preds_target(stock_path):
# get mdjt data ready
	global wp

	stock_features, stock_target = stock_csv_features_targets(stock_path)


	# load the best model so far, using the saved best model by author, not the one trained above
	wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/model.30.best")
	# error on unknown layer BatchRenormalization, if use the following line
	# wp = load_model("/Users/Natsume/Downloads/DeepTrade_keras/author_log_weights/model.30.best")


	# predict with model on training, validation and test sets
	pred_capital_pos = wp_best.predict(stock_features)
	pred_capital_pos = np.reshape(pred_capital_pos, (-1, 1))
	true_price_change = np.reshape(stock_target, (-1, 1))

	preds_target = np.concatenate((pred_capital_pos, true_price_change), axis=1)

	return preds_target

# mdjt_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices/mdjt_prices.csv"
#
# mdjt_preds_target = get_stock_preds_target(mdjt_path)
