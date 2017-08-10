import string
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
print("num of characters in sample 1: ", len(samples[0]))
print("num of characters in sample 2: ", len(samples[1]))
characters = string.printable  # All printable ASCII characters.
token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.

results
