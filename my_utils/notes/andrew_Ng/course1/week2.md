# Logistic Regression as a Neural Network

## Binary Classification

神经网络的两个小特点
- no for loop on dataset
	- 一次性计算所有数据，而非一个接一个地让样本通过模型
- forward and backward passes
	- 模型训练包含有两个阶段：正向传递和反向传递

用logistic regression来解说神经网络
- logistic regression的用途
	- binary classification 二元分类
- logistic regression 对猫的识别
	- ![][image1]
	- Input: 输入值是图片，一张图片的 tensor shape (64,64,3)
	- Input: 将图片tensor转化为tensor shape（64*64*3, m), m 样本总数
	- Target: 目标值，是猫1.0，不是猫0.0, tensor shape (1, m)
	- ![pay attention to notations][image2]
----

## Logistic Regression
- 详细解说logistic regression 的数学表达
	- ![][image3]
	- Input, Output的范围表达
	- Ouput的内涵以及数学表达
	- Sigmoid的内涵和数学表达
	- `w.shape` ($n_x$, 1) and `b.shape`(1,1) 是分开表达，
	- 但有些地方将`b = w[0,1]`vector 的第一个值，见图右上红色字体内容, 不建议这么用

---

## Logistic Regression Cost Function
- ![][image4]
- logistic regression 的数学表达
	- linear combination: $z = w^Tx + b$
	- sigmoid function: $\sigma = 1/(1+e^{-z})$
	- loss function: $L(\hat{y},y) = -(y\log{\hat{y}} + (1-y)\log{(1-\hat{y})})$
- logistic regression 损失函数
	- 损失函数是建立在 $\hat{y}$ 和 $y$ 的数学关系，寻求损失值最小化
	- 最直观建议的数学关系可以是 $L(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$
	- <span style="color:pink">虽然MSE作为损失函数，也能在 $\hat{y}$ 越接近 $y$ 时，损失值越小，但对于二元分类问题，预测值是概率时，并没有优势？</span>
- 针对binary classification 的损失函数
	- $L(\hat{y}, y) = -[y*\log\hat{y} + (1-y)\log{(1-\hat{y})}]$
	- 当`y==1`时，$L(\hat{y}, y) = -\log(\hat{y})$,函数图让 $\hat{y}$ 约接近1损失值越接近0
	- 当`y==0`时，$L(\hat{y}, y) = -\log(1-\hat{y})$,函数图让 $\hat{y}$ 约接近0损失值越接近0
	- <span style="color:pink">选择该损失函数的原因？也许是更直观？</span>
	- ![][image5]
- loss function vs cost Function
	- loss function: 一个样本的损失值
		- $L(\hat{y}, y) = -[y*\log\hat{y} + (1-y)\log{(1-\hat{y})}]$
	- cost function: 所有样本的平均损失值
		- $J(w,b) = \frac{1}{m}\sum_{i=1}^mL(\hat{y}, y)$
	- 在不断最小化损失值的尝试中，`w`和`b`不断得到更新

---

## Gradient Descent
![][image6]
- 公式1: $\hat{y} = \sigma(w^Tx + b)$
- 公式2: $\sigma(z) = \frac{1}{1+e^{-z}}$
- 公式3_a: $L(\hat{y}, y) = -[y*log(\hat{y}) + (1-y)log(1-\hat{y})]$
- 公式3_b: $L(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$
- 公式4: $J(w,b) = \frac{1}{m}\sum_{i=1}^mL(\hat{y}, y)$

#### 碗型函数和波浪碗形函数是怎么得到的？
- 碗形函数的由来：
	- 逻辑上应该是将`w`, `b`,代入公式1，2，3_a,4 可以得到一个基于 $f(w,b) = w^2 + b^2$ 原型的函数，<span style="color:pink">(能得到吗？我目前不知道如何做这个代入过程的演算）</span>
	- 如果演算成立，那么就可以生成碗形图，如下图
	- ![][image8]
	- 碗型函数，决定了convex或者是碗形特征，会有一个最小值
- 波浪形函数的由来：
	- 如果 将`w`, `b`,通过代入公式1，2，3_b, 4 可以得到一个基于 $f(w,b) = sin(w)cos(b)$ 为原型的函数，<span style="color:pink">(能得到吗？我目前不知道如何做这个代入过程的演算）</span>
	- 如果演算成立，这样的函数会呈现波浪型，没有唯一最小值， 如下图
	- ![][image9]
- 在让损失值不断变小的目标下，参数`w`和`b`是如何更新的
	- ![][image7]
	- `w and J(w)`单独构建的函数图，如上图
	- 只要函数图是弧线，不论开口朝上或下，有一个不变的规律: $w - \frac{dJ(w)}{dw}$ 不断更新`w`值，从而让`J(w)`不断缩小
	- $w$ 应该的更新方向和量，由 $-\alpha\frac{dJ(w)}{dw}$ 决定， $\alpha$ 决定变化步伐的大小
	- 当 $J(w)$ 变成 $J(w,b)$ 时，只需要换个符号就行，$d$ as derivative 变成 $\partial$ as partial derivative
	- `w`更新的公式: $w := w -\alpha\frac{\partial J(w,b)}{\partial w} = w - \alpha*{dw}$
	- `b`更新的公式: $b := b -\alpha\frac{\partial J(w,b)}{\partial b} = b - \alpha*{db}$

---

## Derivatives
- goals
	- 不需要懂太多细节，只需要构建直觉就好了
- 什么是derivative?
	- ![][image10]
	- 函数图形在一个瞬间的变化：量和方向
	- `slope` value at a particular `a` value
- 简单情况（线性函数）：不变的derivative 或 slope
 	- $y = 3a$
	- 上图中的函数的`slope`在任何点上或任何一个瞬间变化上（不论`a`向右或向左变化），都是相等的方向和量
	- `slope` 总是 3， $\frac{dy}{da} = 3 = \frac{\Delta{y}}{\Delta{a}}$
	- $\Delta$ 要求是极致小的变化量，所以称之为瞬间的变化

----

## More Derivative Examples
- goals： 更复杂的函数图中的`derivative`会不断变化
	- ![][image11]
	- 复杂的情况 $y = a^2$, $\frac{dy}{da} = 2a$
	- `slope`在不同的瞬间的变化，或量和方向，可以是不同的
		- 当 $a = 2, \frac{dy}{da} = 4$
		- 当 $a = 5, \frac{dy}{da} = 10$
	- derivative的计算，可以借助书和软件计算，推演过程不重要
	- 更多复杂的函数的derivative 的值
		- ![][image12]
			- $f(a) = a^3, \frac{df(a)}{da} = 3a^2$
			- $f(a) = \log_e(a), \frac{df(a)}{da} = \frac{1}{a}$

----

## Computation Graph
- goal: 拆分每一个计算步骤，方便计算forward pass and backward pass
	- ![][image13]
	- 拆分 $J(a,b,c) = 3(a + bc)$ 计算步骤, 构成`computation graph`如图, 来一步一步求`J` or `loss` or `cost`，即`forward pass`
	- later, 可以用`computation graph`画出`backward pass`来一步一步更新`a,b,c`，实现`cost`不断最小化的目的

----

## Derivatives with computation graph
- goal: 展示 backward pass 拆分过程，每一步通过`chain_rule`来连接
	- ![][image14]
- 最终要的结果：每次`backward pass`更新的`dw, db, w, b`
	- $\frac{d(FinalOutput)}{dvar} = dvar =$ 描述当`var`变化一个最小单位时，`FinalOutput`如何变化
	- 上图计算了 $\frac{d(FinalOutput)}{da} = \frac{dJ}{dv}\frac{dv}{da}$
- 当`var`，`FinalOutput`相隔较远，用`chain_rule`和中间值的`derivative`来链接计算
	- ![][image15]
	- 计算 $\frac{d(FinalOutput)}{db} = \frac{dJ}{dv}\frac{dv}{du}\frac{du}{db}$
	- 计算 $\frac{d(FinalOutput)}{dc} = \frac{dJ}{dv}\frac{dv}{du}\frac{du}{dc}$

----

## Logistic Regression Gradient Descent
- goal: 演示`derivative` 如何在最小化损失值的过程中更新`w_1, w_2, b`中发挥作用
	- ![][image16]
- 通过`backward pass`, 推演出$dw_1, dw_2, db$
	- ![][image17]
	- $dw_1 = \frac{dL(a,y)}{dw_1} = \frac{dL(a,y)}{da}\frac{da}{dz}x_1$
	- $dw_2 = \frac{dL(a,y)}{dw_2} = \frac{dL(a,y)}{da}\frac{da}{dz}x_2$
	- $db = \frac{dL(a,y)}{db} = \frac{dL(a,y)}{da}\frac{da}{dz}$
	- $dw_1, dw_2, db$ 是每个参数相对损失值的变化量和方向
- $dw_1, dw_2, db$ 如何帮助更新`w_1, w_2, b`来实现损失值的最小化
	- $\alpha$ 是学习速度，或参数自我更新的步伐的大小
	- $w_1 := w_1 - \alpha dw_1$
	- $w_2 := w_2 - \alpha dw_2$
	- $b := b - \alpha db$
- 以上，是面对一个样本数据，得到的一个损失值，和所得的一次参数更新
---

## Gradient descent on m examples
- goals: m examples, m losses, update m times
	- ![][image18]
	- 上一节，we get 单次损失值和单次更新`derivative`, 即 $dw_1^i = \frac{dL(a^i,y^i)}{dw_1^i}$
	- 这一节，一次性计算多个样本的损失值，更新和`derivatives`
		- $dw_1 = \frac{dJ(w,b)}{dw_1} = \frac{1}{m}\sum_{i=1}^m\frac{dL(a^i,y^i)}{dw_1^i}$
- 计算 $J_{avg}, dw_1, dw_2, db$ 均值，而非单个样本下的值
	- 需要使用2个loop, 来计算上述均值
	- 第一个loop，`for i=1 to m:` 所有样本，
	- 第二个loop， `loop` 所有 $dw_1, dw_2, dw_3, ...$
- 使用loop的弊端
	- 计算效率低
	- 用vectorization 代替loops
	- ![][image19]

---

## Vectorization
- vectorization remove loops and much faster
- `np.dot(w,x)+b` 代替下图左侧的`for loop`
	- ![][images20]
- numpy basics for vectorization
	- vectorization 比 Loops 快 500 倍
	- 尽可能多使用Vectorization, 规避Loops
	```python
	import numpy as np
	a = np.array([1,2,3,4])

	# 计算Vectorization运算所需时间
	import time
	a = np.random.rand(1000000)
	b = np.random.rand(1000000)
	# 开始计时
	tic = time.time()
	c = np.dot(a,b)
	toc = time.time()
	print(c)
	print("Vectorization version:" + str(1000*(toc-tic))+ "ms")

	# 计算Loop所需时间
	c = 0
	tic = time.time()
	for i in range(1000000):
		c += a[i]*b[i]
	toc = time.time()
	print(c)
	print("For loop:" + str(1000*(toc-tic)) + "ms")

	# 250190.813275
	# Vectorization version:0.9486675262451172ms
	# 250190.813275
	# For loop:520.2362537384033ms
	```
---

## More Examples of Vectorization
- goals: transform from Loop to Vectorization
	- basic usage:
		- Vectorization 的简单运算
		- ![][image21]
			- `np.exp, np.log, np.abs, np.max, np.max, np.square` 都是基于`Vectorization`的运算
			- `math.log, math.abs, math.exp` 都是需要结合`for loop`使用
	- logistic regression forward backward pass in Vectorization
		- 包含2个Loop
		- 先用Vectorization替换第二个Loop
			- ![][image22]
			- $dw = dw + X^{(i)}dz^{(i)}$ 代替上图左侧被划掉的部分
			- $^{(i)}$ 代表某个样本

---

## Vectorizing logistic regression
- goals: forward backward pass without any loops
	- ![][image23]
- 关键点
	- X: tensor shape `(num_features, num_samples)` or $(n_x, m)$
	- $w$: tensor shape `(num_features, num_neurons)` or $(n_x, 1)$
	- $w^T$: tensor shape `(num_neurons, num_features)` or $(1, n_x)$
	- $b$: shape `(1,1)`
	- $Z = np.dot(w^T, X) + b$, 如果`X.shape[1] > 1`，那就是同时处理多个样本
	- $A = 1/(1+np.exp(-Z))$

---

## Vectorizing Logistic Regression's Gradient Output
- goals: Vectorizing the last Loop for multiple samples
	- ![][image24]
	- ![][image25]
		- 上图左侧是2个Loops版本
		- 右侧是vectorization版本
	- <span style="color:cyan"> $Xw^T == w^TX$  这个等式成立吗？</span> qcg: yes
		- <span style="color:red"> 不论 $X, w^T$ 谁在前后，`np.dot(X, w.T)`中的顺序必须按照合规来调整 </span>
	- `for i = 1 to m: z[i] = w^T * x[i] + b`
		- $\to Z = w^TX + b \to$ `Z = np.dot(w^T, X) + b`
	- `for i = 1 to m: a[i] = 1/(1+math.exp(-z[i]))`
		- $\to A = \sigma(Z) \to$ `A = 1/(1+np.exp(-Z))`
	- `for i = 1 to m: dz[i] = a[i] - y[i]`
		- $\to dZ = A - Y \to$ `dZ = A - Y`
	- `for i = 1 to m: dw1 += x1[i]*dz[i], dw2 += x2[i]dz[i], ...`
		- $\to$ `for i = 1 to m: dw += X[i]*dz[i]`
		- $\to dw = \frac{1}{m}XdZ^T = \frac{1}{m}dZ^TX$
		- $\to$ `dw = 1/m * np.dot(dZ.T, X)`
	- `for i = 1 to m: db += dz[i]`
		- $\to db = \frac{1}{m}\sum(dZ)$
		- `db = 1/m * np.sum(dZ)`
	- 现在是所有的Loops都被Vectorization取代了
- 但是在训练次数上的Loop是无法逃避的
	- `for iter in range(1000):`

---

## Broadcasting in python
- goals
- example
	- ![][image26]
	```python
	import numpy as np
	A = np.array([[56.0, 0.0, 4.4, 68.0],
				  [1.2, 104.0, 52.0, 8.0],
				  [1.8, 135.0, 99.0, 0.9]])
	cal = A.sum(axis=0) # axis=0 refer to rows; 1 refers to cols
	print(cal)
	percentage = 100 * A/cal.reshape(1,4) #不确定时，就用reshape(,)
	print(percentage)
	```
- broadcasting in python
	- broadcasting: 单个数字或序列，横向纵向的复制延展
		- $100 \to [100, 100, 100, 100]$
		- ![][image27]
		- ![][image28]

---

## a note on python numpy
- never use rank 1 array
- always use rank 2 array
- ![][image29]
	```python
	import numpy as np
	a = np.random.randn(5) # avoid this!
	# a.shape = (5,) rank 1 array
	a = np.random.randn(5,1) # recommended
	# a.shape (5,1)
	assert(a.shape == (5,1)) # double check in code
	# change shape
	a = a.reshape((5,1))
	```

## Explanation of logistic regression cost function
- 深入理解ogistic regression损失函数
	- ![][image30]
	- 损失函数，就是对prediction内涵的解读
	- $\hat{y}$ 要被解读为涨的概率， 接近1就是涨的概率高，接近0就是涨的概率低：
		- 如果$y == 1$, 预测值就应该越接近于涨，即概率值接近1， $\hat{y} = P(y|x)$
		- 如果$y == 0$, 预测值就应该越接近于跌，即概率值接近0, $\hat{y} = 1-P(y|x)$
	- 文字转化为数学表达：
		- ![][image31]
			- 上图中的2个`if`条件表达式，可以转化为一个等式，如下
			- $P(y|x) = \hat{y}^y * (1-\hat{y})^{1-y}$
		- 对该等式做`log`处理，即得到`logistic regression cost function`
			- <span style="color:pink">but 为什么这里要用$log$来处理？</span>
				- log: monotonical increasing function
				- $logP(y|x) = log(\hat{y}^y * (1-\hat{y})^{1-y})$
				- $\to .. = ylog\hat{y} + (1-y)log(1-\hat{y})$
				- $\to .. = -L(\hat{y}, y)$
				- 此刻，我们得到了单个样本的损失函数（Loss function for a single sample）
		- ![][image32]
			- minimize loss == maximize log of probability ??


---


[image1]: https://lh3.googleusercontent.com/KqzCHNE4GoH-8Mgqdh7Y6PQkkR0xDcLyFvZbMTHX8cSDTmHB-0efMYrQe2njCjvGaP86tyZ8s2q3XnQ3nsPp9laAt7YgYpCONNkVm__m6mY_fjquRPFbFNn33hyHxu5m_vw1DYhXCWXrVBnjF8Fgdc4f7zOJATLkWnwjOy-2dqrfbc4u20s6L0H5JleMbThY7iZW2QG_PPqDkIgG0qg4F9GdybM-Ku1O_feYYzHFuieCWci4gV4qFjJmuPx4Y9eAq1P7bUx39_ht6BrNIhy02qDDr4vxWSvI7xzoZdR-HVJhkZrqJWidVrwAAIifMITCTLlB-aow1eB6cSSmCrF2227FklG2xUE8Sw2P6CU1qohuFw4hj5IbUVBfmGXPX_2Gmk61CuJYWUx2eFA5iUE3KnhzwPabhl7enmJmQrTR-l6FDa4Kf5g5zw51Ef7MUuKQNf1F_O4ZZmgh-uPgT93AQx0-G8dPp9Bw6uaXbGkw1XfXQ3XuYDN54uD0stA0Jtj_H9hWLDOeunfOBrbeK7XakoGBDQgvs1jGrpfsBlkqHt-LrVIAUnKqMu8mMIqpjRDvWYvqbHVrZ4qsqexRd3lpasfF3F06g1k44pLg1yYNrtU22uWcvulqF3eu=w1900-h1032-no
[image2]: https://lh3.googleusercontent.com/czHGUpQPJZuJYckHZIyBijbxONm0J9UIfwTls8NnYFib8tnD0Dtfz0IHisZ3lGrjUb63PWU1s4DoJhnYFsquI6R5T2ak9wIsI0ALG5xtIwPhY4bYdV20A5W2gAS7gS-3owM99rR2QZ2qzsmIXTr4sldRT-v9cdt6CYQXTkmEo0CecxnuqgWkb-uNP5TJ0JrT5_eEhwRBeHIsB91bNW7g-1tcdB_tCWfBWjrP_exfhJqrp0EluONR7TLaaiFfjhklEQxYSM3MQS9jj9osoULGT48p-sqj9nbuHWBdEZy2XMjjhfxwyL2DjDcJyHyxCw_DVeC_jI9U-9JP_vlgNCO_w4fh9CNfbHFKwmBARehS_DyCDHEJm1najTQVg_woADiVORgZ8W5zP0lnlJKDtvrh35jmctw6CKbBfft_UjqDBb7XWnb_gvIkmHWC_4YU2AKfXb6nF4OJ5-OmKH_gfe6NhQ5AjR9sXI_se7nWKxSNlAdQQGT5bue_-vI5hgm_--Ckqw4jBOY4MVsz4HmJoKuk6xIKpmA1lk6up6YWfRHttCGLtMlC6EezcU8YSPNJCoWXUw2nG9oo6gK1NMQbgCX0TuxHDUCi1WuAAkFUh-d_e_5e8G1gW3GG2C8H=w1760-h978-no
[image3]: https://lh3.googleusercontent.com/U3h9Ymvloex2Y9fXAdSWBjfFjXsLEHFhf189BkYizOryDNsdpLdKRrzLX4ZkWYHBQEoKJvK67cKvEI9MFOnitzYqfTtWOSCWoptYEAI3HgOK6j8HmCVHOw7iciIRrn-1m0HzJTbVb58NFJtYeAnPSo65lXl_PEDZqWf81UpU9B8jyiBQ7vcm0ixTFUE9MF2q7aVJpRn3hnfRJa1b-0-3rmgBH2kfHfrkENa6WuZzIofJp7d50zIOcJfScpDXfowx4IABPRQiax0yK8Ohrhx-Rj4L-9xC3Bk9AsuzlnPu0nBf3S8P-ZKAnzr1XhIDvr7pmzlp0mxUZAnglFTrvlm0dOL_y0pOx0Y3cYBtnwcEHEhD3TAwQPwtPImMKiyE425K3yZ2CAeo7tiGXd5vnfFEufRl-ld831nWoy1Cju0774A_lwcTyH4jcDxV17HCTNNyDfB_NrCiYfNy0spDOUb8VxYVEg2w2nSRKwVpWV8ncy0YHKrBwnJnbTbRL01w6Tpm8KXTeYc6Xxt_cIYSK1THz2WwPx1xBZJzwpn_oZHJghVuf7hnnBZNspARCV00Cpt_N-kelwiPPHvtJznT95ouAhwxXx6QxXxM7UyksCuB0hODF6WNWv29ThoK=w2094-h1224-no
[image4]: https://lh3.googleusercontent.com/DrBLbnF0qSP4XzAd2AMPpZ3eyiNVTNAj0SipdzeR37uy3VDJChbu4Po1nT66_5aDjQpUD0PRVLVmHPL7TIXtD32VE_vwmv4rI2MmSEuTk-upbhnvGERNHZ8oxSMk4wvkYZrJ61TKcGSPup45xemRDQoLSWFH13aBAwdL0ZgkXNa-t0AU7Sr2kNEpcjeYnJ7SvaZCWJmTtY26oapUs1YFbJ4dtb3PU5qcn-PezH6lD-bROu71GLBw_r7e_9mp4pKYBrqYejG99uwAhe-5SOVWV9IhCY2HsTPY_hvoBYSiYAjBEa-2Nd9Q1yeKnEa8hKhgfBDNrmK_oGy2Yylrco0XhFvvBpW2t5xdGWWb87GGSpNuf81sQutFpVkzXk7YvGGYg9h0atKnIv9wA9RZOXfQQlj28p4XCtQ7wMXOnyJLb8AkmWsCs9BmZjj_lTB8fwdNg9uHoQsil17Dajmc0tbFycVHGVCcrRDmmSwVNwbs7O-zae5DIgbDHIuXNHyQCk-6wWcM5vOyHnQxbbGDGFWBl4BrpBNAfyYAx64R1Zj3kY1RwbXrWeLHgIvL-7xDNy4jQXlD3S7mt5ZB3u-xPyfn_GZK5UHvmPZBXIlHMPASlYWmI2DCMBfAL5Yw=w1918-h996-no
[image5]: https://lh3.googleusercontent.com/o7FZdccz5FzTEb_W5kw6oLnq6x-nOz61paXKEqGoZ36r_eCaLIQRjY8o51okupX9vtY8AE0TXGZSmAfN6u3gin35vbEQV-8Ij28ih9mfmvALP6d8mOCF-nlFfaBwx4nuAtmzU_Y3zAeTa5qyrZD7W-PwaBlm9KNou_aLnIyTcJOqySkVP1k41BG97-nQlucHrMDMqeJ21vsRqKb8lmtSOheJOgIND_AN6L1c-nlnaQMUjq16sUnu29MBuqBkFQoOlD75XAU6IofURaVjxgUqN-A3rXuQQEZFS1rI8ISucaA8KmBq0hMeEjZtZj65dZPvmSRYcWVK_zbzxM4r1qBIRYTAFWfRgHKIrupQ0WMAsvWnyVjHNB_EJDDTy831N-PzwIHPJCgB4velciTKdXPiwLYaGBSX95gh7O384OjwU9DRMMPi32KKSa-ebiN36fQps5VSM5mFRiAvvFZJgAGg67iHs0cPm5p8VtZknMeE8WUEfBDRcFJnKv8xPIKk3QvlwVA1HVdyq9fKDfFh_inDZMBinE_2ySQVqIupjfIHR21yZzZVFZ8gmhaoIqupxqSmB06Z7fWz-9rEFwhEiUOiD_0vsZzuFKr28Wq43XQhlb6zQloD3UJj_Oal=w2090-h542-no
[image6]: https://lh3.googleusercontent.com/ZBqjR9tg54EdEw1iF98ujFqsYDy93_H4fc3AhXUSO0E5a8q9ZIXN-exjtkcQXg-HLmY_NxloCC90OMPrq_bsNWj2O11NpxyWOta7GEl9myWZ3vvDC8xvh2TZz73WHfM6WX9W_oVM5R0B7NaO0PWH2mCWTxPcOHHiFxxcy1lv9SXr8rbpc_frK6pXl7ZL7Qp6zeiubUYY2rryBKpFeFwxc-9AC_ULl5t52bt7gGpZcc2vZ0-kKt5ooaqmX6kQG8r2c9nB-i87_xZh1H5cT2pcuFG9DN5dlO8aoNmGFFLPL1A3dflQDbD-nzgBvIGiHnlM7G2ZM-h7_ICpVt2IwPui9pCwlsNqcDwHnaLDZ3gCyf4_2aPONaR4GIDCtSNWGw5VhMgDO7vL_D3XhsBWYlVS2E_qBXlWkGrOJ7XfPOmxiuDwWTRmYrOqPBe0_EEh0G27pO_e95nr7FMAt0AMXq6lK4fMaORARrAp_QXVRDsVoxovA7Cz0o0PvCdjPKdeygY5etqDwf26_6vsROwZ5Ex_vRjOCaB0HBRL4xqs5sa2yJU3GAA3J1y_YI5U_ffXhKEKSucOWmj8m_xKI5afsZOjP0GKRVOy36nmp59H48eA3qgMzsfeWM-QUURZ=w1914-h958-no
[image7]: https://lh3.googleusercontent.com/y_J_yslGvUEhnDzf0RtQ4zFXrlWG427hUzNL5F1b2WCASmkD1I0NhYXKaDwgt4q2lGT3s0l4IRetZiS4sGCP-layErG7Cldf1ZVe54yjbh_r0Stv58g1ndnq1N6bHpVyeqbCReR3BqX7Y0skmdpxhroB-mP1iNxLQ7zroG7lMDE0DR76KjE95FxokVFqF4lMaNkj0vpSJ974rhjRY7h2M47vdmcL4b4AmK-ygwjr6zXiQUaVRjO_en8GL3AXQtuOr4Sj6iRGqGiYl8uD7KI--t5pqCiudfqnCTzcoXkGIRV1CXkz-8pHEzMfvFCT5BUOodECOEYlRIBWoem9Jxg6Xsh9O08jS6euISCBNS-5jiujN3yWeYscPS-YfZmb_nYidqkTAP8IvIe5-6mAT83FkryJV4o47nnDlrTCb9r91yjqBSsAqO6UhMMsJzTLIUwRZncN9r771dQ2_heASPPQeJzfnC_poENDK3e-mpJ3DV56DSgdMStZOINPM-MnkTCwc7rdNlgoeRQ1BR5P0JH4-1deQtdFG0FjuUpyMeCIyINkvIjpvBn4aXvZuhrasVZOfkYSjkQ_NInBUHzzUTwdEiJYLBAketVYYFx5h-mWy7nmEeyKcN0AYy3a=w584-h322-no
[image8]: https://lh3.googleusercontent.com/RZEdyoheelD-a0NgUbb3ZcAHJqdTz_KXa8aGskYHXjLCKSF8wuM2Jbr6mreNStM3h72pxrtMagNlvGp1leTWD8zquLU_5HiP6lUcLc78RlpR5cgTLi2qIVZ8OAp69PiLdUaY5ng62XsHY9XxjWRMSCrsyWzhY3FMouCI6OVT3dmdwR0bG5EZJqdE-cJq2CnKqnO1C0DfA_MaV_4sfh_edVIEgx8BSPbHqZv7mNssh6MFE7qQ8IwaClrytFzklVM5tT9GZ7fZysWwhoLNoGb0COVwZu7pAZWXH5TKTqfCEIWeajZdQzySNIIZYDz_3zk5Sd9TQGNhj6B3UqAEQKAP5mwbeOaP7fh2Myf8yZFWDVLcqwla7jKoXhdJcZF9MiGuU8L9ry0X3Vq1pfe0Xa8hNFA3jYLq_usqinBXArWZTtsSt6-oPDlwLLWIwlQWKgc7q9haB3sBkW7MesyQPo3v4_QPhGMzLKhei505VwMioEXibA7RjrtB60rn3oGNsOXBJE_SgOlF_vsoQ-vRis7a-JMPBK--rd9CAF_OI6Y5yrayRB5uSXvOEktRLFjhGsxexWao9S8zcbi6hqESD6gesztK70xfrER_NzFIkOwM3GG9O81j9k08=w1094-h946-no
[image9]: https://lh3.googleusercontent.com/atytBnhFGoh80MrvLtXVpJ3QOQlEgrLS03nM7cOGR3aKAi7a4-77xqaAp0WNA_0MjsqSw-NtsKQyvH735rLELyHwUttCYIGxY_RonThTM7nLddkRfjC_7N_PCShgNc9yv1UWTHHtG06moqfwQyuL8zg37f9SJy7NeK3aziGU-mzEy7qIpB2jGHW_4hMdZXuNcbIEZJfH3b3eCT_OYGYrdVEX6xsDvnOLQ2TpBR1Qd-iMp1EP15Ew1ToVBbSTlm9II9p7yCzWgk-K-b3MG6oWq1XIV75NOt6nK74Sh5_36tzNwipHIc3JMtxriedzYq9_1tgI9q9CMyYcuuDoe1FubFO1eGpHsxaaYGgrJSs_cKlK3lldKTmTX24Hrp2jj0EK9WDiMMgkRdWa0kbqmjJuLUNpEPx9aZZ_GrXNX_5jn4tu_sB3ZPGyPoNLqf5F9s9GH0qyO6-Uf0TpBAtzSvv1m_6nO97uNKHSs5jrks37wMjDbiEhlSsJDL3mbmmKyHd10YFEf-1hKyCRSro-o1IhWF0L3WBP4sDVkWpaXreWbqyuVrfJoc2z8E-awzB1nowL8m3OHub589PZU6Zga-4qxkAuUzAAdYfquH3rEs3ohbiR8qytSsqm=w1056-h842-no
[image10]: https://lh3.googleusercontent.com/bUgjlsjYY_eOrxlXmc4jgNDiG70b6h2Uw3etKkeZTiRenLPcDEwfcnAx6EbrrgNOlL9g6bmO2TyPP50o-OB1Jq3S5dWU4wxHehyjNol_ydIK9NpaWY3eMhwqJ09smmSGME_2oEU_gTF_x4g48SNI-fwpn8Ba35f0Sq9uVKwBqlBL5baPfRp-oltV_7ae9dDTKv1YXqQreSbCVI_Vkg7XZKUSRFDE_u9Hl8M25CTe3Dimay1BbjmL8nH4mky8HYxcq8vHtd6s05ybQDL_tQ97xWmc-XwGVj6xx2nU9khY571qB-Bm8lMqXx_Pv_Z9R3sMqpytX-N-gdd1C7UbMRa2O-0VttI-1_z7AlhoTlt2I0WUjogGDC8hIy0vqlDE6w5aTVNqXv7ezS_qqs3ZedCAP-2utbqUE_kSjdZdO6PPtiH3dTzvNUER0_Vci0xLcxDMYsJX24TfKOKGC7s2P9R866lYA1AXY2AfYdMQ_CHcF0YwiuP-9lcHRo-F-cqTLYuf9dqz4bdMB0jrJQuddZUEdsVmYvsTmy_H-Xyx_r-ksnDpjU1iKOm36KQgVUDlpSiq72WTCZjuZEAvhw0s-zTVND5fgtG71to-V6Dq9bVhkkYfuk4jY1tq=w2280-h1224-no
[image11]: https://lh3.googleusercontent.com/IW1Dk4SvsbW6P-VLvvOx6GTA7wD5bjHlKFqB6vISEzH3y0wnRbqfxU5ZU3jYmTSuBKjjGtdrni8JPEZ1ZYGDwyrb87eQofJrO9wEXjvq_am4GajnEdUxhVRNQIK5_tFSv6FyLQNZUjylwHouLhFUx6cQPMg5hR5XNo55bImp1dTVZ2i47SqF11d5ntT2-qvYRK7SfIKhbO73lgN3p0UgGQnNCQQWPYiUV8ke5kzJqmC8d2dg6O_qQOWwWTyIWjVkJyCC9TW_LXuDjMmkftDpVc2plXLTAYdxOkomyEt3CBMB3kCNmboKHPtxW4jzyMrTwdK9fFkWMCZNZoVjXfvRjeW75zKLOOLadForilKZASohGNB6MW9zPq72DYbTfKFnzipUEN0IL-9_-pBzVOTBaxbAREFxs4faRfBv6lFU7RiQgQG3-5hEdpC9S-XIuKlTaE6JpjB1GRvE3dNcQhk4UB2iSw-fy5HWPckqQk33y22HevMH2DbvqLeTnORnO4PA8jzP-zx-bSrP0xIhTWnqF-GWH7Y7UcY54gl__lbe1EQQC-Vt78ruacV2wFyx_dkCbd8fo-VbZf26MqnQ-1H1_6Q0WyKfFzkNuwn-KA8PGjgbiRzp8v1K=w1840-h948-no

[image12]: https://lh3.googleusercontent.com/DVg0vjXhDN09OHeW4BhQPDmaqLxLsy0IXsw3IhIFVhw0IEk662iVLN1-jWqdNxJzGeOp75rCndPvW8MRICTPYP2dpLLz6f6j3ZAYOSvRbyy6q7N8ZjLkQecz_eoYKZz4sj2YZi-3RbKXQE0HKQ4mDbRl_TEkUULPEk7tQ34fKd3alKyAtZMX0iqNcPH-D74Vnbp2fG4_Y68mVKe3V2rV5JKv6wA5AkGEHQS1q0qFO3x5Bj_J_DrMASFfS5IdTidXz-WYBTKVjtwzp1AyS-nO28wm9KDeicOjQz4AuYczx9puNfQF6eHAdRSBHDOEe8ZxLoWYWzR4r5bo7qIK1SCFspL_dH5YYVeVyc2epvLZwHekRtM_nn-9-fG_6PX14f-guIkwnqXv_dgKIW3zLkCUkmS85_negXXzVN6T3fMaBUkJOctVKiBQd-9A2B2A8ZcEtPSsa6ReWrBdbYiY3ZvySF446UWCG03NDu5YBr05Zzg_gqJE8qyrUk1ECs5nd3kdZQo_1_2e6NG4ytOytcsVeXozjf9qWp1rFq9nLtDS9-4YiLNJ6cp6D5ryeG390zqkFwVQ1tVSaMQZiszSVfzvIYfGsKy4tbq8ApTGYnQ5KJs4ui2c2A8X=w1788-h908-no

[image13]: https://lh3.googleusercontent.com/wkcF_rqZDt29URuC2unwhj_WgE3h8QRCCnNFGYB7lbj94wwUAVqzUfuOcKDZbGMmJzk9EP2KtmwbVaG3-LwW73Uqz3O_C9smIIYicRtLcKVVIexCWhSd8BMoeAIHoPii0OkCwxbuF1gtBV3gOrRtdYlEJv5u0ydS2wNnoYB7i4zSkgbeRZsTjCqM05DpIMGIaXM0nKVjYjCuDfDyjDYFxpAbfKCtfPq1uboSbwy5W1ZR2aoGePmVZPNGqKCj6dE8vSI1yzUF9GoBdfcV7scaV8JxZLX2NlWhd79H_odDJfI-eOrAZ2A7rxPz9Tw8DifCxdx4wMHBmVXs5_9TVDR3YKxTpYV6zAe49X7N7QiO33KtrhcgbgvIP6WxQ4GPYLFFCsLbKDWs22cL-mGdp2IvvX0yuH6AnI8K1G5dP1ebnvrqY2CBrDj03UU07YTNThFuUxzQ2mgKdCIWpVtUsZzLfY8gdRK1pyNMV8DVhlF_yL1FgwQEz-a-2wx_41K2tF0o24xZgAtCTjf7OdbMrHViy2xdwq7p-vfCXpV3OIC6eh0MSVEBwNzs0REOorFPtfhfBDpRhepVMYNy4F99Q7y4DE4rzn72DaTIiz6G419UJgE2Qowv-Z0-=w1730-h884-no

[image14]: https://lh3.googleusercontent.com/N9U_-aXjjF186oIZIlRYWAtZLZhMqcikpD7gD6tHprVS-8l4Xh3otZ7dHprpQ_S-68rXv6s0oh0LkfKaRqVJvbK93qdRxeCEnBquY3a-p8R7hbYkBxsNiiYGPBGoNsZYbRopithcmljQqRFFBQvyHe7yrXFqYMsH_-GXhOe-MvHpuGXAkh9msRbUkIX_5izFqLgDrzNDoKiBwKVlYCkTj_hZQgjT7qy1vh3IFqyjeuzBvs81rlMZgpNCYXcZsDT_id2UVfszvMgFeKTr2pNsVMH65pEFbCspgF_Og-YGHG6XS6PTRgxqovl9QripqXLxnIJZDz3fC4Fp6HULWqZqP6S098VM0hax7UvcUo6gAmAjhQNuk5btwlLivj6hMOUr6KL_Xcks2cQKAj7MHS__iIc00crzEt0suNJT6kkRlX_b_ndja150MNaaWy50ZRK95oD-l3bqIINd2Hh48ImyChRKU3e-9NZS2ne4qoKtWWnWUQ2rPpnUD29GvI6mW21OSmrjM5l_kw_K8u03AeAnbDOj69y9OfI0i-zdYrAlZCnwVFrXBE8zvOJ2ss6ywZMVqSfTaovyfAoJHRCThmEd5iIC9uYOSnAYQnqW1AEyj02s3gBn0AnD=w2316-h1224-no

[image15]: https://lh3.googleusercontent.com/RNApLOV7bZfK6Hr3I_ckiGI5DEPj5vfqe9wN2xgI7O_1aGmiRUhVct6eLkyCXBGaQyhy6ub5H58P1DQ28SxQe_gFhrjMTZ1HLLTZHTNAJn4ynu25ceszbd-KRQmilUi4hpl4xnCSQrLfagLbRUub0zOh0CuUeAswKi9zZpnOF2_oZv3IqwgY5yll5ustI9AAK9l4AGOhTsfHQ3RcqT5NIdxAXr2yj6ZEQJkrqcPxWicihcpSyywKMBMkiGDUJf6Smt2BQrs2s8hwIeNDHTYF5ivEVZOUKQMD4LmUHbLGfiKko5HO7G69Vy5y5AQKKJFZTr7xyC9yb2fbeOl-cpdRplIoi8RTQMWvO8QtfEAgijdORngeMQ2L0JUyPOS_KutYdqX48rPKdKi8bcMooh0qoY1_qot_vpJlXxycRVfo0zc5z1Z4Xa34uvKZ7AQAYmoexl2snDGq-P_hFqCyNVCvMYDNE9uEBOkCkwZFBPau53L_o-SpCNjbuJRU8AJ5rdqw_EHfD4BCc1DJmoQIxe0hqXkQS1aGOdWq5t8Xcg4KXKdYpreuiyKSzq0mNb4vtNYr0_zrS8gKG0ODKVH9dQUTOL_2b4uPYw39B1vbO8rDyrTmvn0MxMJM=w1812-h930-no

[image16]: https://lh3.googleusercontent.com/EHN2To4V7l0eh3Rja1-IBxyU-t5e4RMGTVivik9SXhliZPsgNNhrYWHbWWNwwu-SDHNtNxf7593vY1bgqrefqU3R9ydHY2iSi2r-xv24gkJaOE8LDSbUgMovlQwMaBrT-yvnS5I_v9s-tW6d7mYuO640VRPB_ro4-3hEYhfCv-5q-WCofXqrlQvNADJ7mFqhna0M7bRKti9YiDAd9bR0xcbVWXtFEywrCOMYi8Aid9EFAWJcLztb_2yh9IlF3h2E6INsWgPOtiygooySoEma73OeThx4qOfbpoNh0rzaVth5W8fZWY5MAC1eoC4m9jkRmPE5rbsHngQpHnpP-oKBYlPWIXd-vvXe47ym747APLk-ReSUfSw-n9z-V_1DuCoYHw3orAq4Rd8IX0J-W3iHLi6Sr7ZJC0WZmbB5qMHxjziaIXvy93IoEjub7ox2DsPH-TvDZz9SGMuC1DJMEpOOT2guu-cjFaMLu0vxLtVwe5bHWla0PEsp9L2MkO0IAClS0jLXbKrtzJ2tywgCGNd13ygY-mzVPz-_4_XxMsLcVyTdma_mIbLShAEOHEbhSfvoRIahRB1Z4vJgk8PLQ4PD724UNFtWviNvZA5Iw4s5IsM3-MpJvjzc=w1646-h942-no

[image17]: https://lh3.googleusercontent.com/S79dP_KJsqN9-XOkXJVUFY-439VsGa3N4F9DXSAsWM1AyuUmhmBDKCSNEWrUYLy5Q1LQVHVbo3LTHv3im68j0e3R2cQNF-KPAd6tzMGudQn5HPqZVVj4xboCt1XoQsULJMiEQH9gWH6Uj76Jvr76-THucBeQdJGFeJLcVr9cKW4yrmiiOeliG9_0eC9075olQ-fWlC29h5dZMV7u0evxLWr5SX5gf1WRrF1jAKTmx32jBWLfA0S-0ZQPJ9RpA0m4voK5dS2po_lTeVyCNCeQUqXSlWLmgJV3gOFyMjNOnqJx8qo4eyXCQn4Ki-ZN0LtOrZSoUdKoQrMUJ0cPMCYItxh0_xgxbeB8tC9myi5NUydmCLqlFToSwT-CeDN_yiiwJLoN1Jfb3gVhINPyc2tVNagw2o-WjFZp9i-onjT-qg5qwFEsveAx-5SmJDGh2jojBOdGHe7aZ4Mrs0AceROASe4TZChhAO-8UBWXBBVNIcdN99Gu5UDlTaRIXgJ2oGc6noxVWsdisozQ3MPsXL_4QfpAsNc1op-yyQ-04glcAOb3hzK-7cusZBqeybR1qKErO2W7CskaYqjG0jdtWhcRx9bZd6_-MUrb1nsZ3rj8V7Fzne5WGF5B=w1866-h930-no

[image18]: https://lh3.googleusercontent.com/Tw4H0JSDUsspF7jGyyqWzqGY_toH4XqnYorSNwsFdgI57rXaTqwR79shf3DDVGbt3Hp6IFQCHrk5jysn88G7n-thxctT1HwBVlYl3F1FkF8HosWyoI6tR0I8bTJXE4dv9iV_rJpNg0S4fc8ObvvDrpcDeekJYGjwYmxn7U2nKird33es8ekmb8J9DA4x6TSlqn1cJxGJH5rqD5c-EXkYO-XeFscDwDIr0mxpb7joiM5MCkSTojK0qnbPqYIyx-JBCSVXsKFQO3OHdAWA50sNQo0aDbTgdgiePN7ZEkoyFHQbiiVz4t3Whg2SS69wff5f8IzJu0O5S69t-5i-ibbYmOLN6Ra0ww1i9zcUUmKrnJrQKa64LV7cwH5IjNeHaclJzNvAs4JMc2bEo3s-y-MrQ5dlGPq0C9_qnhym8Ak0hmbqtGIQHPJ97eBJkGj5MdBE_D871AIEX23SWUJODYj0D-sqouZO5f8PGkacTPNS9k2keifYCzQxAzYwjBfuy78rwykir6dNxU3tW0eXGOq1cIKEBw6AhpJLpA1eybtI53QnzJLAEKqUSAAzLhqD-bzDB9bdDMoYa6X9Vd4e6RP_CWiMz_WlxY-AH9WpDtamhxvir_QQsIA2=w1888-h786-no

[image19]: https://lh3.googleusercontent.com/oeFaKKc7oxihPTQnlQhqB6cWqdIXFd3uFiUQ7JzeyFUDhn0bN2p-2aZiQMmFqEj86RD_k97Xv6p8e6LXWu03DIUY1YJsC5r5RTNL4Bc7EZWSzpBD35k85YceL4LCwBTDHGhLKq4PcefPgbV6hvRpQrh0NUINNQ7-VU8kJtzJxq2eZ4zF6z4Rgrsm5CtP71X7V1SWJOuWFuYtBE1ztUBF_qTaFmUBFbDYn9Ap9jib6Jaj7uMVgea1NO3GmLbq-a7tVtz984huCoevSP5UsctUJ-T9NmLTdUYbH7gZVyJ4HgBkHQSIdO0m4rhTNVzixy9NRzAbJaaAy03SWJNizwkdIdtS9xbzcHA7-mIfili53pTIE2CoEXQfPpcXdwvRnGCtz_O5QbPsGG5UpxEUaOQPw0CMpozPIcWRaR3wXeoOkcUO2i0-s9Zfkm-O8v4a3Y8e189ZYEplAIRJhPlUnp4L8E4B_KTr_YPlB0eigOe6Kx6pJaBZJ46AxcnMwIqijQRGDO7fX8a9L035GmlUxvdNLf66zbJys5QF4U-135CmUxAVHIx_7VeAhJ6ZHwKybdXRYFrKdz76l5KNT85Pt1pRJd0SL_tN_Bj2hE6h83gHhcGuw3rg6GnR=w2066-h1224-no

[images20]: https://lh3.googleusercontent.com/N0gi0s-ACbES3rxQvgxHz828ntGJqcS7tWpgtCTXCEZY467UJ2Tvu7k3U-f71Yu72SieE0yqX0OYKcCZFulDWiGwQzmrN0sfKnSzO43ZesLbcYczq_Te5KLPPo3Q6b6aP8qrEPEJ2HisdJGCKMWk0FFqytFAIEd5RrHt5BsrZZ242f6Ujbk1_mx9iCdIM7MrO-3dI4nNx3nZpJJOr2wu4T4kEf_2j7rAP_aEKSYDj9KEuMRFSjpNgAg6pt0Wqjq6qBsabTeVhYtMYUwBQbp3Ts-bADxg-48CoatoKwLH-p7DPK40zYwUm-vxIaYaJa7g7bmmM-_yaag5dGmI2xNDTihZiFJuyvYzmp3_JHw-XARNTFhNFC5jahou1xzTFLxHRd-pmgys55-7SDQ87Ka4U_EdjUGrCR72bjx8yi-JiKe95k6Udz_bqqlUA6mp2WPv_kcwWgXr2f-7a-UVJO-pw-inDoKC4-ffYLOmr7llb_LQWxal8_p88J7p4WjekFR6p1VsV_Tn7FBKcsEWYNmVHTOl15Y0io3fS1H9gBQYZKcVmsF8vucua9Ub9yP4_FrXa8jRJsQbTSSPtvE39Pee0zKEIfs6x4wbYtKHgpGBo8qFRsj_VMJz=w1720-h806-no

[image21]: https://lh3.googleusercontent.com/TWVRqBqJFS9Wus4ON9XhR1aF097QrFv2_rdX6WdR33czMaBamOZjR7kOvxb-e0fR6iNUzV9AD6wkVAjx0GcWP8FDfD3-EJYDyKUF9qOeqFFyVrvdOPsUpZYZljjTKTX_U6ebVL2KBP27yXsQ7-TB4XSvOsPZpbcwZUSHw8GTctzv2eiISHt5qMyDawwDyH0CXQOSvH4ImDCObbO4aFLdXUPzrX22We9RnFFgdy9xc6pUE5bvwWKSawaMsv7QY8TIeV6lk9P_gr5wH8QQHO-jEWLMie4pxu62kI69BElvmOUkS8QC4MX9jFesiz5hs6uZa_YV-csvWrpyCYrjaMVVSVmRKrotWCWqk_GOBeUIB8i1N-DfEUDaBG9Pe-N4iYrvUqvGcfV-QIeAKrE6Rk_CXrhkKFECSQqgNhvEF7l00_M3Ib65iuhLqx2zLkQVTb8-u5MS9DDLx9qU8JjEzUsTkOzXpr0scO1IYOnNNIIV2EfxrFU14EZccON0QMQ9HG0ftQK5rTGxKChCsrteKQObQ-37of0VFwb4wrDLhoVC2YkoERHUbM0I-aE_1ZgEVd_9K1FTGdNXpEFDm2UU0hbmlAhXJqQqmPbaht75CJCMukwasSb5majO=w1692-h928-no

[image22]: https://lh3.googleusercontent.com/hNn5Ox0VAcXDwHRDZhMiCY8yOjcxs-K44yClG3uoGl1QvgJLr86Iv902ifOCE1fcoWcl-g_IevAeIby1kN1tICJSaHWlBCsMyQH0hxlJNEs_LfqYC5WolCec0dkw8fcKE73zlVQ45tEtPmZJmndLOjvwCJts8uPG09M44OSXiXNvk1eGy6Iue0sH-NiSDVKmmdDp1c3gr75ytQUmCNLgLAS7lBqKBGsaD_5EkwcS1LBBuJYTITzvJiOF5N-qXNO5JnqSZ5nqMFXadjbykbjN7f90yQIEFTpvd-ZJwRmpbYNbjxsFt2iSx9feVIMwK9VSauFJxx-Nez6m1aJK0KWxE06EM_6YKbhlhw2p7vEVUZzqnW7_cXfot8DWM5k_p-wN141PvnoMf887NlURD0hlzxuRvygl8klovcfRMGrj0LZZQUCWGoYd-VGhduoFv3MnKdQ88CwrGiNArEClKwTi4yk48hvBHyEMBbxzmhpFFJhVDMKLhw7Xs_hsVihR-a4VyWxvcWsENeLfzlUtiE3TdWkNl0EUDnWf0aPkzprBy3rUcvQ0PWZak0eZ90SS9u-p_8qkoYJ6JVJWhCijivx9yI6B7Rz1XlexztEhKtT9q1ZHMFOrHB-D=w1786-h922-no

[image23]: https://lh3.googleusercontent.com/A_Oozm_tp_Ixx2TQCwmq278Ob-XdG4NYXDM5P0vxqNI3l1lJhC8SXt-GpeAvbUl437rYYR-Om0weAFBpuo2j_lkjCrjZfnT0-dCd2DbpnyxGnKUtBLQE0KB4EkHumZFhk_EnfsMRdFp5256s0B6bls9vhqcnAvXVcblh2OWOcCVLU8U4m5c7fnpcjL9KEIwZs0CkqkZATnvXkmVvIphTK8S_fwpJQCoqyLj8GETIlOue_lfU77hEotOkZWa_z142Jm_x7OlsqVS7b-bcKvAOUiCAP8P_g_ok-E-RlM_JOe_pz-ADRko7fw82KdosDUvEEvgj_JWMZIT6gY0KpDxHC-UbOP4sB-SOj6xXAJOTxLRCKrzG7YFJ2Wa7wKZ8qcWPUm5pcsWryLl6HSOw20UZwoU-b4pC8dR168QjoPSEYImiF-smSYtMgdH7HWkijWZ1o_eh2pPkemJDcrPdDUo4sw8cZ_yzdwWNpZTfjV560RL7lfNHhMfCgZKWWHZSpmW8ym9jS9D1ySmddUn65zpMslFZiii9OzlC7s9aZ8GRguRzDVajLwmy3yNznEZX0vEvTB5d2zPNerJ496NPJsx2rHESlOlzihWv9-fDzQvLVxxMQb_VgK-H=w1728-h922-no

[image24]: https://lh3.googleusercontent.com/Q9Qst-lyae3_iMLNGcLNRWdIwt5oeq506FFgbV1DteXCQehYyhH5S9uCa1_on8YBWUWXMlpZGniX61NGAEqDRGlDPG24DrTH9qtHII9pSe_9zbIuFqEicI1gTK3ElK9oY46aZ331iit67pLkgOJ9X9Tzz9-VIhGx3-otrdVKrxAk0Sq4jCSZaQVUJcVRHSC98hbCEkSVsLlJ_V8fFnnVgayFKHCEqfZmN1aD3b-jNVYcqv2DusCEVzckHmBFVEMxsAybo5Wvo7gD-Vw66mXOzUsNSsxBtqTy6hg6NOZs-cV8BwY10wBqM3KRpvqneJVcmePuzhiyF8zW6-F5tMrQwjybzJs_9jBB7tI6Uic48yC7Sxlar6oXtnCkn16y2OvmHoo6WiSGv8OhCRGjLTyJyv5DeElvX6vvAdS-LmnqhSMpnRPQD0jzCROv3iVvj1TrK3gWtEQxFCMUvu1oSnDCsZr40XQtvMMw5sdfiusDvjJUx5QiD06iygaCDlmlYSgf233L8wpf9MuI55tz98I__BQt5YG5u-GbbKcBvyI8ejgOoy_zkCBysTo8GSSDzmhaMgz6NZ5pZgjrk9r6kPyT4i8_2MAVMJkZidWdDC-ZW_Xh575s5avk=w1836-h940-no

[image25]: https://lh3.googleusercontent.com/Bpjpj6UjP6u1cWB8vs4c6ZM4bK1pCsq2xR6IEYBFgoxS_ZRW5OzfkHWaEpeq7YZGcRvW_TSgLY1LabqfE9sgpguUGqhRmpkVucKVqW4TpmqOvzhZIB6UIegiQOFIZ-PXpsxYAGo1HrKCUhEsBW-d3RjQdfo0p9avoZZScDBvRA_qw4nY8UytsqWLY5aT8x70ty_ztcXxCPH8bgneepq7HGnyd04OxiFwubz1jQ91p4PUBzyeAw9JT1LpNXs87Fdl_Q5y3QzZnzg9gKcA2M3aoyxYW8fXwDTsFvjxpl4N2qsXGxxQ8haCILGJNrjXD2pDQXFLQUbfMNGgPiHZO3cB1AdVdR05CDaVwv47AjqvGeJtuqervMEekuQ4eQlu9yWW0sbnKl1O0WmbiQBum45KOqaI6cOK3tJYWw73lrWyxMy-gkuldan2IdI0ncA13NaqhLIBh_WfIPOF_lPxYrmpf93u202-V1Bd4lBEBsk4ycxiCOMBMf85dtV_gX_vyQJQ1QyvF-nJnns3jXZ6v8-F6xkNkmVRUtTmxMqoBJwwACnHfSB8CPij9nyPbXlYEIXGcwcCbwF3K43bDOwC3zQkOM1q6zlhh1plvPw4Gm5UzvkTWWLwceRf=w1814-h938-no

[image26]: https://lh3.googleusercontent.com/G97-8yEM9cHSml7lDesCtw-zkqw4UKGsjXTGo4Xp_nnnQV_-g3X1Oi6118SqCT4XbN2yv_zi2HdoSfPFso-VWmGYxQG_69ReOnr4sLM-Uw5Sn7l5t_qbnwmVSeQpZz4wjUFlPFtHJgcCSCVK8h3dXLaVhgwzE4zrBiGVdOp2bX3b6m7znPY2YdicI9NT3lgqCcpsHNlghGBzaBxioPLh6bU38CzW5wuSXNp4Te1_0PLu6mQxoO0xP8qOArgchPLWK7lCOhQYmLXyb4UWbj4_vKLAODXyCswm09d2rFAxdra8BKsP4K95k7kVQg6399tYPgP1Cs6JA4k89u5cHkh7uza1m36t6HzVYaX5PkozMdeqTF2wdfQEgy2nYYEUoSsurzsEEaQpscSBwThz3oI7Em34eWR9N36aPQrQZNGVSs1ZEaGor6d3T6Tkgss3cpl_6RwfT5nVNtWNGWcW_WVaolj8nk00lFRdinWbkWi0kCJ36-rdhgy7NW888EbOJPB-lpLGwWWfpDYhW82ueMhgmhbarmYW1xOW12YPW_-hlr8clQ3Grwc1O-vlzTtPJaT1U51FBHXwUSiQd549hZuXt8AGoaSppEnqoUHSa43yptcPpqX5ktdU=w1834-h762-no

[image27]: https://lh3.googleusercontent.com/_LiwilVzfcqIGWOO_OqD2WSenJ1gtTbMhXiINBOuXMRj_w8Pli1nEEIMmnw3IOXkyLDcGB1FuL2JnQOYy9D9-Sv0KVz27_9R1r7D4830H3p03XzIkTKULTQbB4JC6zLuOQlBjpTcAgQU3xjKt3TC0P0umuXMvk59O0XFSUYfG61MyD-LtsnUdQVrYIbzso2hnVslUbH0D9YhsFPOsJQLJQy4i2YHpBMhB5wG6gdjttJlCAd0KqbijvB4nPWR0o-J0ySGU9hcSJEo4Jb7O8RW4nrjrzxA9r_mttc3XvrsjhkprCd5p-2WHgheWBYfBFP_qvzJVCCjXRDfN9MnSZ2HDtx0I9jb7tNbtjUu4ExNSoHd5eDRVIaswNnWblH9rdoPdCsIypyeWP6DQ02AoFTfUy2m3oEG9mfqAXrzeY6E_sabPeXPbf0Tl5Ir5EA5VX-feft38I1MfnR-mVS2FAWlBOFT7lapwN4jsg0t6QzJtYGmvLggiXdhATolYAS7xtcO0oeY4S61JDDkTB-bwEOMFWuVIJqgxABMUwNnBV_qlh64NqkZZ4qDz5bxQzfoHWhHMdnC8adhRceQjViyizwJuzznU0NLeX-7rHCqyLhEdC0iIkGjGzDD=w528-h291-no

[image28]: https://lh3.googleusercontent.com/G0rY9_-DYu2L8T13gZtozgpwcKPtrMNg5Sylvov6TRFmtFfKyB9vjuS0aLW3dpwSiDmAN8W1cPiZdvOAgrKXjeMUTnCvuxloACWkrNW_0CREkbZJu3Ih-pXvKLHfBk8qyiX5zUhB2-2aeM_KrC8TvyYqSjrnK2YeoN7EOKqouDOKsLKO6P5ZAEmKCb2KVzFpc5gXTH0Y4L8NcfX17hGVRf6JvEBZDHSbkIK4yKDrX08rAOLBOMWteCXWDQkEeWFFHlTj9H-Rkh0ar8g4MvdFiO3LsoSCKBVtDYXd_q4azW6UvlYv0_U3HcOVEax1dsg-Je8fVIjL4hKVjzImgT5D9D0GLDPjKCPDXJNa_iQ0-3RsPSwCL-5cFYDoxwHfuW6Z8o5ynxOzotxCf2AIqdJMUy81dcr4JyhlWY_tqK1dUqZ2_bP7qKn4g_DZkKAM9ayO_Mq2FnDjrlITameV_fSEXpDWdAYmDrOKxNJoAsmArvQWXvcOC6RKMFCN0wW_1KVUt9PiR5E5nafCwKPGZXgpu-C8_L0hPrYv3eINTTK7y9eCyw1XyhxucR33uDQivttDCdWrhXVuystZMDSTiK9ZqnsmJNGv1NpSI48HCZZwhbWbpFLIUp7B=w1708-h968-no

[image29]: https://lh3.googleusercontent.com/nzBXX42obHFyAs2aRoiKEinpqqicNrJfbvTfe4L72x-KYrJYcjyTuMT7qPf4xd6pF7ohi5MtrkkDqj6CdwbLTvxlV1WRSNkvnTTgAaZuqMYoq1R9NydleB7JWOTVtCat0YjvXyaCaZg0wn29dXlY-Pm3OKvwLWqtrD3YrNya4z_iV_6iVn2cvc6RSMRSuhFuSwxfn0e7Tz2S0_hzYrfB_wL6Xj9d4mW3vdaEWv1xdsNWsIfwY1ky2FanoxVCVSKKIws2wsRk2IrkBDwi-_plV1Pv-DoO_naa2urfe9_MA4RX9EGrGYDAIlI4-DBeJRa6v3fP5YhDVpPXMbYQuMasnCDVaw6KDc9n8teSXbqEWtMdgthuEcJCIL12JrCYibePKs7dCliLPI-_mLmysLBFJxg6MiIGjlTCfb34C2r13BRiTs3d8V4Gt8z8sHqa3XuFQ8JD7lBRtC7XbIfZ5ww98jQpCJK6C43LCP-jB5De373Aud3xNVcQKAPvTO5CkLAzy6oomNNlq-TaexyMxNvXKeaR1Dz_rPPj3ts1xlE05j59UE53afvg1DruvmETTKqtVbyhxfCLhIstQeP1HcbV9jyLCiQ116lmBMg0ShqUf3MO4f9wB9ox=w1886-h938-no
[image30]: https://lh3.googleusercontent.com/0D_-4x2pyC3PichME2jKaR9BIIYB2wdEYBabec6I47lcmk-AiT63gHMAjGqXswEUr18l7FERmn9X6PRMnrDDXEix0skyl-fksuvtHsoCAhsR449U-ZSyeYhSjyzVq0YO-ZoZC4F9ixgAjNYBM3kbscn_3DxWiLczVJ7udEQ9kOXPu6H-ksB148fTI06NVTnYB2HrLkUB8FYj-4FwZzEG6Wc4moDaTXCPBxbZh8Ta_X7vDXCgDZjL-iB1x58onV02vM8M8z8alz5klkQY6nkAtI1yJULl8e1M2QnSjO2cTiM3qfMDP_VKaXFjlI8IkRBaULQAiWh_TWOLEIy9Ac-eY0yGy2WgGif3EiADiJ4kRP-USiuoD4N-J4GEr-CYD9auV6cf97t4ZPkBV7yf5yFp9omvDFhsL00Ua4dcXs9AB_5H_soRM1z8gNz5n-CyWeoOsj1ltv_lybY3Y1fsy6n3ACKIlsqsyqeTFFhRdVbJE3Vt7Xm6_S_QlsIpD2WZDGzrB5yijFyIz3mCSMij4_2loILHIpPaDWhaAkXijtAPC6ANSyGQUSiSleb8FBQq42RsZaBQVC5JQiAKNYLXIt1QaOJlV7OJKeUlBRbMc33QpFIgnqzy9Z6-=w1532-h734-no
[image31]: https://lh3.googleusercontent.com/G2qAfy8Sp8pXW928xYJlf5_o9Cnd8RCS53RrTju_aXMjJTsw8LxPRnBcfGx93mawGiaPv-QCWvKWM0fQucTr3T09UDI2fHYrcsHUCRUKz-7Zf05feg4qh0Uam9pROdE86MEEIxdkhRgyS7u1f8sYai7vUTr1MJPDsUAJfHy05mhf913TJZIhrl75a_bdNofNsINHBCwvbuDTof5A2MTDpDi2jM3iEt8S8qHGlg4X-mDMpN_9mT6QnrIk5LHyzE1ZvuVzaExrDYTZ18UqcYn0xkjiGdjhC8Y3sCktm75IEHKciiFepWkmNf_qZQL0KsqBPDTFEYzZI2ToivZKUUORLGM_ZG1kyWrV0GYzf-n54IkmX7nSiSWinftX7iQ65_VlIBcODeYL8yosVxRiREuY-gzZ3roR7nnCThj2IC4WXMTw4BRvkytXRGNH5UPqkS-1j6QAZKWsra-TkkEQVwRxjKCd72Cyj01t1HwtiPVqqQOX5VjQidd1mkdRAnzeHnQT5lW1UOXuIuhjW409EnrP3wDRVlzNRcCOyeImL0QNhs02R5IYMA6zlEHZ4R72FfP9GmlZXP9AcNu7wNOmu01TuQn0PgP3gWmwCDmWny3SxrtO2_XIphUw=w1914-h1016-no
[image32]: https://lh3.googleusercontent.com/nF-KYY1OSyLiBSdq2sjTwWn1K2NnoFqMaH2Yz0HLaPnaYEcMw6Fi4w1Ns4l9NnhfC9IxuiJvuZR-37v9eeeGShH_nm3EDxyLyhv_coB4WN8Emmu38k6uL_7faVX78ltFyrF3--NybsuRdTTGsl0PZv2DAk6Jh7v15yRwQMSwU3YF2sGTRK8mCO6Wz8jgrsbSqj-2dyMowWb5NG-ZAJ8Vtlwugh6t7obYAHKNKoQs0JbMiy7bsmYd8q-Do236T_0sdgczPRPuoZlXECCHrSPoLyhzfW2KxINJgULmq3qlwhjDK19yeXcylnUyIMy1rEln4qF2Lpuh9GPM58LRn19Vf5u4Hz4n1YpwZNt3ATe6PW_-gYvIa1rHl5zIepi4AkkB9sVPxpXj1La2xaM3ZBgqPwjI6i0cPHUCA26BqZHz9vZkjX4sgGmABdIUJIixhEk2fR7tBZu_0QEyRoaLC23a0bhYSBcJfcm59WWppoiwqpxk4vCEknIOJaC1z688rJgw-TVSZ9sDELQ_P6BZIpWPWEs32isHKRXb0muqycm8nRk5BRyNa1dJwyPr8ieFEyD7fx7OCw6eFG_TEtv1l2o3-juAQ0wWOCnxQHakNDxDHTBRt4tLu1eu=w1846-h974-no
