# Week one

## goals
- 哪些趋势驱动了深度学习的盛行, 深度学习在哪些领域及如何被应用的

## objectives
- Understand the major trends driving the rise of deep learning.
- 深度学习是如何用于监督学习的
- Understand what are the major categories of models (such as CNNs and RNNs), and when they should be applied.
- 深度学习在什么情况下能和不能高效运转

## intro
- course 3 是deeplearningai提供的独特价值

## What is neural network
- 房价预测和单个神经元的神经网络，
	- 房子面积（输入值）与价格（目标值）的关系
	- 该模型是在模拟RELU函数
	- ![][图解1]
- 房价预测和多个神经元的神经网络，
	- 房子面积，卧室数量，邮编，小区富裕程度(四个输入值)与价格（一个目标值）的关系
	- 每一个神经元都可能在模拟RELU函数
	- 我们无需考虑每个神经元代表的特征值处理（如家庭规模，超市距离，学区质量）
	- 我们只需提供足够多的数据（输入值和目标值），模型自动刷选创造特征值
	- ![][图解2]
	- 三个神经元通过调节，四个weights的大小，来自动创建3个特征，至于特征是什么完全有神经元控制的参数weights决定   
	- ![][图解3]

[图解1]: https://lh3.googleusercontent.com/4PFbPtzypvkR7S0u_JID0BJAAPPVNz0GIJYmnAxfw8f0-33y_R-YyoPfFHJk2R5ZjQ6JSZV2V85qdfGzyMRxMzONBo_4Hjmm_S7ERKa081AQGeAmx6hV4YayVgyNCOU4FtAkKwcf8Zb2pN8ylC6-wHjtL4ApAFb9r-faa2ly1PUqvgJ47xiFppyUGX3pTSjm6-ino4Ivac3E2Yhmr_8Y3nRSI5cqaSQofoA195XKlGZkg5iPobThaOj8FE_0otm68hDegmVAFIdoBVn06-j6-1QIQQW10oEt7rUdRy0KpiEmvEWFQyjlDXUTImhBRoN2JQ4pFxu4VAiL7PQVTwKLZqXrx8yqFC921rsYr69LxjbtRagmbpUvJ-0QzO6AHzCnPdsQkHnS-z7dNDPrCyZf4J188HU2pDmajjoonV8lVSRxt-7esVpb6stLqe3FB1C1OANpUvGENIC5JmjSNST7StsdnL_o1uZIf-MrbimrZCYzBGgq7TjgomWehiM8LJ1tKpwde7r7QB82WrrqDHWAGs58gOr2hHP1vh3Rl-04HTI03cnmMerMQ6db2Bq3kGw2CUKdsMko87Y_0ZFAPjMnYLxhLMgChIhvOrr95NvGztJcxh9QM-8gMjMF=w1738-h928-no
[图解2]: https://lh3.googleusercontent.com/BglrkaFXAvK_mlj1uBFSDIeMV44Wp1DMF8ThsekgcRuVdI2-QwcYWqvYetrUYhcdcJv-hWZIDTBAI7oOP_3ZBh5YeWadWF7fhDubiUjYGemTeXC2lp1lN2p0qa9OFo0AoXkLwV9m37o0Powug9CAPJeW5FhIA7wm1HhBDpniN0F8N7h-WL2jvDczrRt_q9VjxnD3IxoSnBxFhzaxR1aPP9DaB_t1ZwwrV47lzQbZ4oIjYszxzjcg5e02hgAcvHT35AY-x3h0tVLuCoITcTaG5jtiBAZAurillWqfOHTOACXj9lC5hgHW4iQSd574ahJF1DzOoZm4849BZHFZacMShZbynOzo0w3-aXNoMrJddjlzv2JecTtlJfxOhBmu0YF_CyX98tlShfd1xAak67HMcsMmKBpIaErWLrnrmWCAQbN0uQANxSOWdGrYuOGnCKBMgP9acD6mkgA3VFFY3NjVHhQyGvlQuvto-QFlzTaTg9u4DQol2kZz4wXLMryetXgQpNIkRtvAdUdYVGF3H-e9KNR5WvCGbBx5RVU7HCQSpJk7WHSI0AFp45FgXKcn1ZE6UPn43eHhmSSecjuVLemMaF8etUYwkJzlG-nmzBv8Obp5eWfD56dzit3O=w1478-h860-no
