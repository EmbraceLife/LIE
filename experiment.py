import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20)) # 训练特征值
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10) # 训练目标值

x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.


### 简单的说，这里 input_dim = 20 和 input_shape = (20,) 是一个意思，通常我只用到input_shape, 而shape_dim是简单情况下的表达

# 实验 input_dim and input_shape
model.add(Dense(64, activation='relu', input_dim=20))
# model.add(Dense(64, activation='relu', input_shape=(20,)))
# 1000 是样本数，无需代入模型结构；只需要用到特征值的个数，即20；
# 有两种表达，`input_dim=20` 或者 `input_shape=(20,)`

model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

## 验证
print(model.input) # 验证 以上两种表达方式的效果是一样的，因为输入值的纬度是一样的，都是 <tf.Tensor 'dense_1_input:0' shape=(?, 20) dtype=float32>
