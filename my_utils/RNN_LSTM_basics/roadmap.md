# Reading deep learning with keras

## RNN chapter 6
- RNN and 1D convnet 的特长
	- 处理长篇文字，时间序列，连续性数据
- RNN and 1D convnet 的应用
	- Document classification or timeseries classification
	- estimating how closely related two articles or two stock tickers are
	- decoding an English sentence into French
	- classifying the sentiment of tweets or movie reviews as positive or negative
	- predicting future weather at a location given recent weather data
- 做两个代码研究
	- sentiment analysis on the IMDB dataset
	- weather forecasting
- 本章会学习哪些内容
	- pre-processing text data into useful representations
	- 什么是RNN，为什么有效，如何用来分类和回归
	- a bidirectional recurrent network, 和 leveraging network statefulness to process very long sequences
	- 1D convnets for sequence processing 及在何时可优于RNN模型效果
- RNN在处理文字时能做的事：
	- document classification
	- sentiment analysis
	- author identification
	- question answering (in a constrained context)
- RNN在处理文字应用的本质
	- map the statistical structure of written language
	- simply pattern recognition applied to words, sentences, and paragraphs
	- computer vision is simply pattern recognition applied to pixels.
- 为什么要将文字转化为numeric tensors?
	- deep learning model only accept numeric tensors
	- Vectorizing text is the process of transforming text into numeric tensors
- 将文字转化为numeric tensors的方法
	- transforming each word into a vector
	- transforming each character into a vector
	- transforming each n-gram into a vector. "N-grams" are overlapping groups of multiple consecutive words or characters
- N-grams
	- Extracting n-grams is a form of feature engineering
	- a powerful, unavoidable feature engineering tool when using lightweight shallow text processing models such as logistic regression and random forests
	- RNN and 1D convnet need no feature engineering
- 两种tokenization方法
	- one-hot encoding of tokens
	- token embeddings
- one_hot_encoding_words
	- working code
	- 将句子样本转化为基于单词的3D tensor
	- tensor shape (num_samples, num_words_per_sample, num_vocab_total)
	- ![see the tensor][tensor image]
- one_hot_encoding_characters
	- working code
	- 将句子样本转化为基于字母的3D tensor
	- tensor shape (num_samples, num_characters_per_sample, num_characters_total)


[tensor image]: https://lh3.googleusercontent.com/qyV6lhrpBBlom5l4liapUwlYcZIVMMqvXKhWE4Bz8ZeJW1HgxmUyioiENDOHvEq3-R2Jn_RGpgLjfH_KNYTlmv5OhPvjeAD5eE1sVxsfthMAQ5noWVD1yh-KrQnNal5uK6crNPDfGZoYZQTy2Y2uHCs4xcGCB8fJMlpQIjcfCHYMBIr6UHS7N-GlX5gaeqMjOoVO-aljs7V7xvk0F3moiM7_opxHlxzBtGBmxA-T-HbRqcBCBUnZL-waXB1PCy0507jcsWckQFMjYV5qSMiHYgeOguCJdMuJB5xFCMSHhd0tbXZH3U9R3A-O40YkGwi-q5sObCz8SRPRFCTlpOUfUELI5uggTIm4kPwWkWoq2uR86u4rRdB4IFnNYdMhZ4kbxfrqsIK_NHbhbcbKKORW8Sk0bGSt0PvnshTXBut3Pmk6agpSTf4uZQE54kflur9_hxliaas3RhqYdezvwcFNPLXq5BuAi0tBQeze4pJdTt5l_DLlaMw90hY3va3KsrxORoD_ezhInUjiPvWuU2EWLUJU3SnZVkhrWG8Yw9aCnfJ95krska_kxa51xfesCJ9R7zB9zpaKo2vtu-7s1TRv9IRVZyzmf9REZukiQaJUUkaUDXXKsshElSH9=w958-h722-no
