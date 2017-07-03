'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from tensorflow.contrib.keras.python.keras.preprocessing.text import Tokenizer # use tensorflow api
from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.keras.python.keras.utils import to_categorical
from tensorflow.contrib.keras.python.keras.layers import Dense, Input, Flatten
from tensorflow.contrib.keras.python.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.contrib.keras.python.keras.models import Model, load_model


BASE_DIR = '/Users/Natsume/Downloads/data_for_all/word_embeddings'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
TEST_SPLIT = 0.1

########## What inside embeddings_index? # first, build index mapping words in the embeddings set
# to their embedding vector, # in other words, store downloaded embedding set into a dictionary {word: word-dimension-values}

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) # 100 words vector dimension, 400000 is word vocabulary
for line in f: # open the file and read line by line
    values = line.split() # each line has two parts
    word = values[0] # word part
    coefs = np.asarray(values[1:], dtype='float32') # word_dim part: 100 dim
    embeddings_index[word] = coefs # save two parts into a dictionary
f.close()

print('Found %s word vectors.' % len(embeddings_index)) # there are in total 400000 vocabularies stored in the file

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)): # loop through every 20 categories or subfolders
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index) # set id for each category subfolder
        labels_index[name] = label_id # create dict {category_name: category_id}; there are totally 20 categories
        for fname in sorted(os.listdir(path)): # loop each of 1000 samples in each category
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1') # open a file of 1000 files
                t = f.read()
                i = t.find('\n\n')  # skip header, which is useless part
                if 0 < i:
                    t = t[i:] # keep only the content of a sample text
                texts.append(t) # append it into texts list; therefore, len(texts) == num_total_samples == 1000*20 == 19997; each list element is a sample's content
                f.close()
                labels.append(label_id) # align each sample with its category_id, store each sample_category_id into labels list; therefore, len(labels) == num_total_samples == 1000*20 == 19997

print('Found %s texts.' % len(texts)) # (np.array(labels)==19).sum() 1000 samples in each category

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS) # def __init__(self, num_words=None, filters=\'!"#$%&()*+,-./:;<=>?@[\\\\]^_`{|}~\\t\\n\', lower=True, split=' ', char_level=False, **kwargs); # num_words: how many vocabulary to use in this model, the downloaded and used file has 400000 vocabularies to offer; ### tokenizer.fit_on_texts(texts) create many useful attributes: # self.word_index: dictionary {word: index}, index: from 0 to 174046, 0 is highest count, 174046 refer to lowest counts; # there are 174047 unique words in all samples together; # self.index_docs: dictionary {index_word: counts_doc}, key is index of word from dict self.word_index, and value is num_docs this word appear
tokenizer.fit_on_texts(texts)# texts: a list of strings, or generator of strings; Updates internal vocabulary based on a list of texts; # tokenizer.document_count: num of samples processed so far; # tokenizer.text_to_word_sequence: convert a long string to a list of words; # tokenizer.word_counts: dictionary, {word: counts} added up in each and every sample; # tokenizer.word_docs: dictionary {unique_word: counts}, each sample count unique word only once, add up if appear in a different sample; # self.word_counts.__len__(): total unique words in all samples; # self.word_docs.get("the"): how many documents or samples have "the"; # self.word_counts.get("the"): how many times "the" has occured in all samples; # wcounts = list(self.word_counts.items()): wcounts is a list of tuples (word, counts); # wcounts.sort(key=lambda x: x[1], reverse=True): sort the list from highest to smallest counts; # sorted_voc = [wc[0] for wc in wcounts]: get a list of words sorted by counts from highest to lowest
sequences = tokenizer.texts_to_sequences(texts) # Transforms each text in texts in a sequence of integers (each integer refers to a word); # Only top "num_words" most frequent words (top 20000 most frequent words out of 174047 unique words will be taken into account. Only words known by the tokenizer will be taken into account. # sequences: a list of 19997 sublist, each list has less 20000 unique but most frequent words `for sq in sequences: np.array(sq).max()`

word_index = tokenizer.word_index # total num of unique words in all samples; also total vocabularies based on all samples here
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)# Pads each sequence to the same length (length of the longest sequence), If maxlen is provided, any sequence longer than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or the end of the sequence. Supports post-padding and pre-padding (default). # previously, maximum length of each sequence is 20000, now maxlen is set to 1000, then we can check data's sublist length won't be longer than 1000; # data.shape == (19997, 1000), for each sample text, there are 1000 most frequent words to summarize it

labels = to_categorical(np.asarray(labels)) # each sample text has its category, from 0 to 19; # to_categorical convert 0-9 to one-hot encoding
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0]) # indices of all samples
np.random.shuffle(indices) # shuffle the indices
data = data[indices] # shuffle data samples
labels = labels[indices] # shuffle labels (one hot encoded)
num_test_samples = int(TEST_SPLIT * data.shape[0])

x_train = data[:-num_test_samples] # split train and test sets on features
y_train = labels[:-num_test_samples] # split train, test on labels
x_test = data[-num_test_samples:]
y_test = labels[-num_test_samples:]

print('Preparing embedding matrix.')# use downloaded embeddings_index to fill embedding_matrix for 20000 words, although there are 170000 unique words in all samples

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index)) # between 20000 and 400000, select the smaller value
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))# create zero array with shape (20000, 100), num_words to be 20000, EMBEDDING_DIM to be 100; # for each of 20000 words or vocabularies, we use 100 word dims or features to define each of 20000 words;
for word, i in word_index.items():# word as each of 400000 vocabularies; i as each index of 400000 vocabularies
    if i >= MAX_NB_WORDS: # select the first 20000 words or vocabularies
        continue
    embedding_vector = embeddings_index.get(word) # for each word out of these 20000 words, we get 100 dims or features
    if embedding_vector is not None:
        # replace embedding_matrix (20000, 100) zeros with real 20000 words' 100 dims or features
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words, # only use 20000 words out of 400000 vocabularies
                            EMBEDDING_DIM, # each word has a vector of 100 dim
                            weights=[embedding_matrix], # only take weights of words (20000, 100), out of (400000, 100)
                            input_length=MAX_SEQUENCE_LENGTH, # max length of a sample text, 1000; a text is no longer than 1000 words
                            trainable=False)# keep embedding weights fixed

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')# (?, 1000) build placeholder for texts with any number of samples, 1000 in length each
embedded_sequences = embedding_layer(sequence_input)# output tensor is (?, text_input_length, embedding_dim) == (?, 1000, 100) == each sample text has 1000 words to describe, each word has 100 dims to describe
x = Conv1D(128, 5, activation='relu')(embedded_sequences) # (?, 996, 128), 996== 1000-4; 1D convolution layer (e.g. temporal convolution). This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. # maybe, Conv1D is like a single vector or one line of 5 pixel screener, moving from left to right, and from row to row
# conv1d_t1 = x
x = MaxPooling1D(5)(x) # (?, 199, 128), 199=(996-1)/5
# maxp_t1 = x
x = Conv1D(128, 5, activation='relu')(x) # (?, 195, 128), 195=199-4
# conv1d_t2 = x
x = MaxPooling1D(5)(x) # (?, 39, 128), 39 == 195/5
# maxp_t2 = x
x = Conv1D(128, 5, activation='relu')(x) # (?, 35, 128)
# conv1d_t3 = x
x = MaxPooling1D(35)(x) # (?, 1, 128)
# maxp_t3 = x
x = Flatten()(x) # (?, ?) not (?, 128) ??
# flatten_t = x
x = Dense(128, activation='relu')(x) # (?, 128)
# dense_t = x
preds = Dense(len(labels_index), activation='softmax')(x) # (?, 20)

model = Model(sequence_input, preds) # a number of samples as input, preds as output tensors
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.summary() # see layer output tensor shape and total params of each layer # model.layers[2].count_params() # calculate each layer's params

# pp model.trainable_weights # display weights shape of each trainable layer
"""
Important Tips:

Contemplating the following two tips deeply and patiently !!!!!!!!!!

Compare model.summary() and pp model.trainable_weights, we can see that how Conv1D weights or filters are used to screen embedding_1 tensor

In fact, for all layers, either weights of Conv1D, Dense, or that of Conv2D, consider them to be filters, to screen previous layer's tensor

How embedding weights transform input_1 (?, 1000) to embedding_1 (?, 1000, 100)? deserve research later
"""



model_path = "/Users/Natsume/Downloads/data_for_all/word_embeddings/pretrainedWordEmbedding_2.h5"
if os.path.isfile(model_path):
	model= load_model("/Users/Natsume/Downloads/data_for_all/word_embeddings/pretrainedWordEmbedding_2.h5")

model.fit(x_train, y_train,
          batch_size=128,
          epochs=1,
          validation_split=0.2)
        #   validation_data=(x_val, y_val))

model.save("/Users/Natsume/Downloads/data_for_all/word_embeddings/pretrainedWordEmbedding_3.h5")

loss, accuracy = model.evaluate(x_test, y_test, batch_size=len(x_test), verbose=1)
preds = model.predict(x_test)
preds_integer = np.argmax(preds, axis=1)
