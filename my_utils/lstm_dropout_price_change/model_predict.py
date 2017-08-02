"""
predict_on_loaded_best_model
predict_middle_layer_output_on_loaded_best_model


- get new dataset to predict on
- load best model
- make prediction
- cbind predictions with y_true

Uses: func: get_stock_preds_target
1. load the best trained model (Note: BatchRenormalization can only be loaded using wp object)
1. convert a stock csv into an preds_target array

Inputs:
1. stock csv file_path

Return:
1. array: preds_target

"""


import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras import backend as K
from build_model import wp
from stock_csv_to_features_targets import stock_csv_features_targets
import numpy as np
# from renormalization import BatchRenormalization

def get_stock_preds_target(stock_path):
# get mdjt data ready

	stock_features, stock_target = stock_csv_features_targets(stock_path)


	# import BatchRenormalization or global BatchRenormalization won't help load_model()
	# only wp.load_model() can bring BatchRenormalization into globals()
	# wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/best_models_trained/model4_new.best") # xiaoyu trained this new model
	# wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/floyd_model2_1000/floyd_model2_1000.h5") # floyd model2 1000 epochs, but lr = 0.002
	# wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/best_models_trained/model.30.4_bluechips_indices.best")
	# wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/best_models_trained/model.30.best")
	# wp_best = load_model("/Users/Natsume/Downloads/data_for_all/stocks/best_models_trained/model.30.best.bias_removed")
	# error on unknown layer BatchRenormalization, if use the following line
	# wp = wp.load_model("/Users/Natsume/Downloads/DeepTrade_keras/author_log_weights/model.30.best")
	# wp_best = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/qcg_train_only_ma/qcg_model_2000.h5")
	# model_path = "/Users/Natsume/Desktop/best_model_in_training.h5"
	# model_path = "/Users/Natsume/Desktop/floyd_revised/during_best.h5"
	# model_path = "/Users/Natsume/Desktop/best_model_in_training1.h5"
	# model_path = "/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/sigmoid_version_300_20170719/during_best.h5"
	# model_path="/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/lstm_sigmoid_valid125_test250/during_best.h5"
	model_path="/Users/Natsume/Downloads/data_for_all/stocks/floyd_train/lstm_dropout_price_change/during_best.h5"

	wp_best = load_model(model_path)

	# predict with model on training, validation and test sets
	pred_capital_pos = wp_best.predict(stock_features)

###############################
# preds (1, 0.75, 0.5, 0.25, 0) situation
###############################
	# pred_pos = []
	# ### if preds has more than 1 col, then make it just 1 col
	# if pred_capital_pos.shape[1] == 5:
	# 	pred_capital_pos = np.argmax(pred_capital_pos, axis=1)
	# 	for idx in range(len(pred_capital_pos)):
	# 		if pred_capital_pos[idx] == 0:
	# 			pred_pos.append(1.0)
	# 		elif pred_capital_pos[idx] == 1:
	# 			pred_pos.append(0.75)
	# 		elif pred_capital_pos[idx] == 2:
	# 			pred_pos.append(0.50)
	# 		elif pred_capital_pos[idx] == 3:
	# 			pred_pos.append(0.25)
	# 		else:
	# 			pred_pos.append(0.0)
	# 	pred_capital_pos = np.array(pred_pos)
############################################################## end


	pred_capital_pos = np.reshape(pred_capital_pos, (-1, 1))
	true_price_change = np.reshape(stock_target, (-1, 1))

	preds_target = np.concatenate((pred_capital_pos, true_price_change), axis=1)

	return preds_target


def get_middle_layer_output(stock_path):


	stock_features, stock_target = stock_csv_features_targets(stock_path)

	##############################
	# when wp is needed
	##############################
	# global wp # must to make the following line work
	# wp = wp.load_model("/Users/Natsume/Downloads/data_for_all/stocks/best_models_trained/model.30.best.bias_removed")
	# preds1 = wp.predict(stock_features) # normal final output range (0, 1)
	# plt.hist(preds1)
	# plt.show()


	# no need wp here
	model_path = "/Users/Natsume/Downloads/data_for_all/stocks/best_models_trained/model.30.best.bias_removed"
	# model_path = "/Users/Natsume/Desktop/best_model_in_training.h5"
	wp_best = load_model(model_path)
	preds2 = wp_best.predict(stock_features)

	# access last second layer () output in test mode range(-23, 25)
	out_last_second = K.function([wp_best.input, K.learning_phase()], [wp_best.layers[-2].output])([stock_features, 0])[0]
	plt.hist(out_last_second)
	plt.show()

	# access last third layer (dense layer) output in test mode range(-2, 1.4)
	out_last_third_dense = K.function([wp_best.input, K.learning_phase()], [wp_best.layers[-3].output])([stock_features, 0])[0]

	# see distribution of out_last_third_dense
	plt.hist(out_last_third_dense)
	plt.show()

	# apply K.sigmoid to out_last_third_dense, and plot distribution
	dense_tensor = K.constant(out_last_third_dense)
	sigmoid_tensor = K.sigmoid(dense_tensor)
	import tensorflow as tf
	sess = tf.Session()
	dense_sigmoid_out = sess.run(sigmoid_tensor)
	plt.hist(dense_sigmoid_out)
	plt.show()
	# plot dense_sigmoid_out as lines
	plt.plot(dense_sigmoid_out)
	plt.show()



	pred_capital_pos = np.reshape(dense_sigmoid_out, (-1, 1))
	true_price_change = np.reshape(stock_target, (-1, 1))

	preds_target = np.concatenate((pred_capital_pos, true_price_change), axis=1)

	return preds_target


"""
Example
"""
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index000001.csv"
# #
# index_preds_target = get_stock_preds_target(index_path)
