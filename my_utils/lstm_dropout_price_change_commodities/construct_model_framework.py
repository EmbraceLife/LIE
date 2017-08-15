"""
deep_trader_model_Sequential

Uses:
1. WindPuller class
	- __init__: build a model and compile
	- fit: to train
	- evaluate:
	- predict
	- save
	- load_model

"""
import tensorflow as tf
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout, initializers, Input
# from renormalization import BatchRenormalization
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.models import load_model
from keras.initializers import Constant
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.backend.tensorflow_backend import _to_tensor

#######################
# 如何构建自己的激励函数和损失函数：将下面两个函数直接插入相关源代码文档中最方便
#######################


# 将很多函数包裹在class里，方便输出，可忽略
class WindPuller(object):

	# 构建class时，会用到的输入参数:
		# input_shape: 简单例子我们用了(20，)， 这里时LSTM，我们要用（30，61），下面的表达式用的是 input_shape=(input_shape[0], input_shape[1])
		# lr: 学习步伐的大小
		# n_layers: 隐藏层的数量
		# n_hidden: 隐藏神经元的数量
		# rate_dropout: 当输入值经过Dropout层后，得到的特征值纬度（30, 61)中有20%的权重被设置为0.0
	# 创建WindPuller时，必须要为input_shape赋值
    def __init__(self, input_shape, lr=0.01, n_layers=2, n_hidden=8, rate_dropout=0.2, loss='mse'):

        print("initializing..., learing rate %s, n_layers %s, n_hidden %s, dropout rate %s." %(lr, n_layers, n_hidden, rate_dropout))

		# 使用 Sequential() 构建模型
        self.model = Sequential()

		# 构建dropout层，每次训练要扔掉20%的神经元，输入值的纬度（30，61）
        self.model.add(Dropout(rate=rate_dropout, input_shape=(input_shape[0], input_shape[1])))

		# 构建几个LSTM层
        for i in range(0, n_layers - 1): # 需要建几个

			# 每一层LSTM
            self.model.add(LSTM(n_hidden * 4, # 要 8*4=32个神经元
                                return_sequences=True, # 每个time_step（共30个time_step周期）都要产生一个输出值,所有我们有 （none,30,32）输出值, none 是样本个数，这里不考虑，所以用none
                                activation='tanh',
                                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal', bias_initializer='zeros',
                                dropout=rate_dropout, # 扔掉输入层神经元个数
                                recurrent_dropout=rate_dropout #扔掉循环层神经元个数
								))
		# 构建一个LSTM层
        self.model.add(LSTM(n_hidden, # 8 个神经元
                                return_sequences=False, # 每一层只有一个输出值，30 个time_span, 但只要求每个神经元推出一个输出值，所以输出值纬度 （none,8)
                                activation='tanh',
                                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal', bias_initializer='zeros',
                                dropout=rate_dropout, recurrent_dropout=rate_dropout
								))

		# 构建一个简单的输出层，只输出一个值
        self.model.add(Dense(1, # 只有一个神经元，只输出一个值
                        kernel_initializer=initializers.glorot_uniform()))
        # self.model.add(Activation('sigmoid')) # no more sigmoid as last layer needed


		# 选择一个学习算法
        opt = RMSprop(lr=lr)

		# 给模型设置： 损失函数，学习算法，度量效果的函数（准确度，即accuracy）
        self.model.compile(loss=loss,
                      optimizer=opt,
                      metrics=['accuracy'])

	# 模型训练function


	#
    def fit(self, x, # x: 训练特征值
				y, # y: 训练目标值
				batch_size=32, # 一次性使用多少个样本一起计算
				nb_epoch=100, # 训练次数
				verbose=1,  # 是否打印每次训练的损失值和准确度
				callbacks=None, # 是否使用其他预处理函数
            	validation_split=0., # 从训练数据集中取多少作为验证数据 0.2，就是取剩下的20%作为验证
				validation_data=None, # 或者另外使用独立的验证数据，None 是没有另外数据
				shuffle=True, # 是否每次训练前打乱样本顺序
            	class_weight=None, # 目标值的权重设置
				sample_weight=None, # 样本的权重设置
				initial_epoch=0, # 从第几个训练开始训练
				**kwargs):
        self.model.fit(x, y, batch_size, nb_epoch, verbose, callbacks,
                       validation_split, validation_data, shuffle, class_weight, sample_weight,
                       initial_epoch, **kwargs)
	# 保存模型
    def save(self, path):
        self.model.save(path)

	# 调用模型
    def load_model(self, path):
        self.model = load_model(path)
        return self

	# 评估模型： 查看最终损失值和准确度
    def evaluate(self, x, y, batch_size=32, verbose=1,
                 sample_weight=None, **kwargs):
        return self.model.evaluate(x, y, batch_size, verbose,
                            sample_weight, **kwargs)

	# 模型预测
    def predict(self, x, batch_size=32, verbose=1):
        return self.model.predict(x, batch_size, verbose)

###################################################
# 实验： 简单尝试上述每一个functions and class
###################################################
wp = WindPuller(input_shape = (30, 61))

# 调取训练集和验证集数据
from get_train_valid_datasets import train_features, train_targets, valid_features, valid_targets #, test_features, test_targets

# 查看模型结构
wp.model.summary()

# 训练模型： 训练时，训练数据和验证数据的总数会打印出来的
wp.fit(train_features, train_targets, batch_size=32, nb_epoch=1, verbose=1,callbacks=None, validation_split = 0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

# 评估模型训练效果: 输出损失值和准确度（如果设置了）
out = wp.evaluate(valid_features, valid_targets)
print(out)

# 用模型预测: 输出预测值值序列
pred = wp.predict(valid_features)

###################################################
###### 实验：dropout在输入层和LSTM层上产生的直接影响
###################################################

# 获取中间隐藏层的训练输出值和预测输出值
# 实验结果：
# 0. 如果所有层都没有dropout设置，所有的0.0的输出都是固定的随机数量，第二个LSTM的输出值中0.0的数量是0
# 1. 对输入层做了dropout处理后，LSTM内部再次做dropout处理，不产生额外实质变化，第二个LSTM的输出值中0.0的数量是0
# 2. 如果只有第一个LSTM有dropout设置，其训练是有少量0.0的增加；第二个LSTM的输出值中0.0的数量是0
# 3. 如果只有第二个LSTM有dropout设置，训练和预测的输出值中0.0的数量仍旧是0

############ 总结 #########
# 可以只在输入层做dropout处理，第一个和第二个LSTM做与不做dropout处理，没有什么区别
################
#
# wp = WindPuller(input_shape = (30, 61))
#
# # 调取训练集和验证集数据
# from get_train_valid_datasets import train_features, train_targets, valid_features, valid_targets #, test_features, test_targets
#
# drop_test0 = K.function([wp.model.input, K.learning_phase()], [wp.model.layers[0].output])([train_features, 0])[0]
# ##### in train, there are 30% neuron output to be 1500 items of 0.0
# drop_train0 = K.function([wp.model.input, K.learning_phase()], [wp.model.layers[0].output])([train_features, 1])[0]
#
# drop_test1 = K.function([wp.model.input, K.learning_phase()], [wp.model.layers[1].output])([train_features, 0])[0]
# ##### in train, there are 30% neuron output to be 1500 items of 0.0
# drop_train1 = K.function([wp.model.input, K.learning_phase()], [wp.model.layers[1].output])([train_features, 1])[0]
#
# drop_test2 = K.function([wp.model.input, K.learning_phase()], [wp.model.layers[2].output])([train_features, 0])[0]
# ##### in train, there are 30% neuron output to be 1500 items of 0.0
# drop_train2 = K.function([wp.model.input, K.learning_phase()], [wp.model.layers[2].output])([train_features, 1])[0]
#
# wp.model.summary()
#
# """
# ## input layer, LSTM layers all have dropout
# (Pdb++) (drop_train0==0.0).sum()
# 2623398
# (Pdb++) (drop_test0==0.0).sum()
# 819280
# (Pdb++) (drop_train1==0.0).sum()
# 5500
# (Pdb++) (drop_test1==0.0).sum()
# 2967
# (Pdb++) (drop_train2==0.0).sum()
# 0
# (Pdb++) (drop_test2==0.0).sum()
# 0
# ## try it again, see some variation
# (Pdb++) (drop_train0==0.0).sum()
# 2623169
# (Pdb++) (drop_test0==0.0).sum()
# 819280
# (Pdb++) (drop_train1==0.0).sum()
# 3398
# (Pdb++) (drop_test1==0.0).sum()
# 1965
# (Pdb++) (drop_train2==0.0).sum()
# 0
# (Pdb++) (drop_test2==0.0).sum()
# 0
#
# ## input layer has dropout, no other layers do
# (Pdb++) (drop_train0==0.0).sum()
# 2621587
# (Pdb++) (drop_test0==0.0).sum()
# 819280
# (Pdb++) (drop_train1==0.0).sum()
# 2696
# (Pdb++) (drop_test1==0.0).sum()
# 1539
# (Pdb++) (drop_train2==0.0).sum()
# 0
# (Pdb++) (drop_test2==0.0).sum()
# 0
#
# ## input layer and first LSTM have dropout, no other layers do
# (Pdb++) (drop_train0==0.0).sum()
# 2621482
# (Pdb++) (drop_test0==0.0).sum()
# 819280
# (Pdb++) (drop_train1==0.0).sum()
# 3401
# (Pdb++) (drop_test1==0.0).sum()
# 1135
# (Pdb++) (drop_train2==0.0).sum()
# 0
# (Pdb++) (drop_test2==0.0).sum()
# 0
#
# ## no layer has dropout
# (Pdb++) (drop_train0==0.0).sum()
# 819280
# (Pdb++) (drop_test0==0.0).sum()
# 819280
# (Pdb++) (drop_train1==0.0).sum()
# 2619
# (Pdb++) (drop_test1==0.0).sum()
# 2619
# (Pdb++) (drop_train2==0.0).sum()
# 0
# (Pdb++) (drop_test2==0.0).sum()
# 0
# """
