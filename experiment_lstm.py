
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout, initializers, Input
# from renormalization import BatchRenormalization
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.models import load_model
from keras.initializers import Constant
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.backend.tensorflow_backend import _to_tensor

# 将很多函数包裹在class里，方便输出，可忽略
class WindPuller(object):

	# 构建class时，会用到的输入参数:
		# input_shape: 简单例子我们用了(20，)， 这里时LSTM，我们要用（30，61），下面的表达式用的是 input_shape=(input_shape[0], input_shape[1])
		# lr: 学习步伐的大小
		# n_layers: 隐藏层的数量
		# n_hidden: 隐藏神经元的数量
		# rate_dropout: 当输入值经过Dropout层后，得到的特征值纬度（30, 61)中有20%的权重被设置为0.0
    def __init__(self, input_shape, lr=0.01, n_layers=2, n_hidden=8, rate_dropout=0.2, loss='mse'): # risk_estimation, risk_estimation_bhs

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
                                dropout=rate_dropout, recurrent_dropout=rate_dropout))

		# 构建一个简单的输出层，只输出一个值
        self.model.add(Dense(1, # 只有一个神经元，只输出一个值
                        kernel_initializer=initializers.glorot_uniform()))
        # self.model.add(Activation('sigmoid')) # no more sigmoid as last layer needed

wp = WindPuller(input_shape = (30, 61))

wp.model.summary()
