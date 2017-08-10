from tensorflow.contrib.keras.python.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.python.keras.utils import to_categorical
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# We create a tokenizer, configured to only take
# into account the top-1000 most common on words
tokenizer = Tokenizer(num_words=10) # 1000 或者 10
# The builds the word index
tokenizer.fit_on_texts(samples)

# This turns strings into lists of integer indices.
sequences = tokenizer.texts_to_sequences(samples)
# 每个样本的单词数量没有超过10个
sequence_matrix_1 = to_categorical(sequences[0], 10)
sequence_matrix_2 = to_categorical(sequences[1], 10)
# You could also directly get the one-hot binary representations.
# Note that other vectorization modes than one-hot encoding are supported!
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# This is how you can recover the word index that was computed
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# 下面两个tensor表达的是同一个样本数据
sequence_matrix_1
one_hot_results[0]
