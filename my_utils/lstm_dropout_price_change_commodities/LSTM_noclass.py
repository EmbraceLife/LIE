from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout, initializers, Input
# from renormalization import BatchRenormalization
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.models import load_model
from keras.initializers import Constant
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.backend.tensorflow_backend import _to_tensor
from keras.callbacks import TensorBoard, ModelCheckpoint


### 调用数据
from get_train_valid_datasets import train_features, train_targets, valid_features, valid_targets #, test_features, test_targets


# input_shape: 简单例子我们用了(20，)， 这里时LSTM，我们要用（30，61），下面的表达式用的是 input_shape=(input_shape[0], input_shape[1])
# lr: 学习步伐的大小
# n_layers: 隐藏层的数量
# n_hidden: 隐藏神经元的数量
# rate_dropout: 当输入值经过Dropout层后，得到的特征值纬度（30, 61)中有20%的权重被设置为0.0
shape=(30, 61)
lr=0.01
n_layers=2
n_hidden=8
rate_dropout=0.2
loss='mse'
# path="E:/Deep Learning/Comm/OutPut/my_model.h5"
path = "/Users/Natsume/Desktop/best_model_in_training_comm.h5"


#### 训练模型保存地址 locally
# best_model_in_training = "E:/Deep Learning/Comm/OutPut/best_model_in_training_comm.h5" # local computer training
best_model_in_training = "/Users/Natsume/Desktop/best_model_in_training_comm.h5"


# 使用 Sequential() 构建模型
model = Sequential()

# 构建dropout层，每次训练要扔掉20%的神经元，输入值的纬度（30，61）
model.add(Dropout(rate=rate_dropout, input_shape=shape))

# 构建几个LSTM层
for i in range(0, n_layers - 1): # 需要建几个
# 每一层LSTM
    model.add(LSTM(n_hidden * 4,  # 要 8*4=32个神经元
                    return_sequences=True, # 每个time_step（共30个time_step周期）都要产生一个输出值,所有我们有 （none,30,32）输出值, none 是样本个数，这里不考虑，所以用none
                    activation='tanh',
                    recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal', bias_initializer='zeros',
                    dropout=rate_dropout, # 扔掉输入层神经元个数
                    recurrent_dropout=rate_dropout #扔掉循环层神经元个数
					))
model.add(LSTM(n_hidden, # 8 个神经元
                return_sequences=False, # 每一层只有一个输出值，所以输出值纬度 （none,8)
                activation='tanh',
                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal', bias_initializer='zeros',
                dropout=rate_dropout, recurrent_dropout=rate_dropout))

# 构建一个简单的输出层，只输出一个值
model.add(Dense(1, # 只有一个神经元，只输出一个值
                 kernel_initializer=initializers.glorot_uniform()))
# model.add(Activation('sigmoid')) # no more sigmoid as last layer needed



# compile model
opt = RMSprop(lr=lr)
model.compile(loss=loss,
                optimizer=opt,
                metrics=['accuracy'])

# 模型训练
model.fit(train_features, # x: 训练特征值
            train_targets, # y: 训练目标值
            batch_size=32, # 一次性使用多少个样本一起计算
            epochs=1, # 训练次数
            verbose=1,  # 是否打印每次训练的损失值和准确度
			# callbacks=None, # 是否使用其他预处理函数
            # validation_split=0.0, # 从训练数据集中取多少作为验证数据 0.2，就是取剩下的20%作为验证
            validation_data=(valid_features, valid_targets), # 或者另外使用独立的验证数据，None 是没有另外数据
            shuffle=True, # 是否每次训练前打乱样本顺序
            class_weight=None, # 目标值的权重设置
            sample_weight=None, # 样本的权重设置
            initial_epoch=0, # 从第几个训练开始训练
            callbacks=[TensorBoard(histogram_freq=1,
                #   log_dir="E:/Deep Learning/Comm/OutPut"), # 画图保存
                  log_dir="/Users/Natsume/Desktop/log_comm"), # 画图保存
                  ModelCheckpoint(filepath=best_model_in_training, save_best_only=True, mode='min')]) # 训练时保存最优秀的模型，并非最后一轮训练的模型版本


model.summary()

# 调用模型
model = load_model(best_model_in_training)

# 评估模型： 查看最终损失值和准确度
out = model.evaluate(valid_features, valid_targets, batch_size=32, verbose=1,
                 sample_weight=None)

# 模型预测
pred = model.predict(valid_features, batch_size=32, verbose=1)
