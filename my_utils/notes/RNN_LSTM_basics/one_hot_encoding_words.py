import numpy as np
import tensorflow as tf


# This is our initial data; one entry per "sample"
# (in this toy example, a "sample" is just a sentence, but
# it could be an entire document).
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# First, build an index of all tokens in the data.
token_index = {}
for sample in samples:
    # We simply tokenize the samples via the `split` method.
    # in real life, we would also strip punctuation and special characters
    # from the samples.
    for word in sample.split():
        if word not in token_index:
            # Assign a unique index to each unique word
            token_index[word] = len(token_index) + 1
            # Note that we don't attribute index 0 to anything.

# Next, we vectorize our samples.
# We will only consider the first `max_length` words in each sample.
max_length = 10

# This is where we store our results:
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in enumerate(sample.split()[:max_length]):
        index = token_index.get(word)
		# i 代表 样本， j 代表 每个样本中的单词个数，index 每个单词要用0和1来表示，多数为0，1的位置就是该单词的index
        results[i, j, index] = 1.

# 现在的results就是将2个样本完全转化为数字后的样子
results
