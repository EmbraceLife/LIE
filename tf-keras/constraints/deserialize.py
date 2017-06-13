"""
",".join(c for c in tf.contrib.keras.constraints

Constraint,max_norm,MaxNorm,min_max_norm,MinMaxNorm,non_neg,NonNeg,unit_norm,UnitNorm,deserialize,serialize,get

"""

from tensorflow.contrib.keras.python.keras.constraints import Constraint,max_norm,MaxNorm,min_max_norm,MinMaxNorm,non_neg,NonNeg,unit_norm,UnitNorm,deserialize,serialize,get

# source code: the same as deserialize as in activations
# usage: take in string name, return object or function of the name
mn = deserialize('max_norm')
UN = deserialize('UnitNorm')

# see source, get is built upon deserialize
NN = get('NonNeg')
NN = get(NonNeg)

# serialize is not working 
UN_name = serialize(UN)
