"""
",".join(b for b in tf.contrib.keras.backend.__dir__():)

abs,all,any,arange,argmax,argmin,backend,batch_dot,batch_flatten,batch_get_value,batch_normalization,batch_set_value,bias_add,binary_crossentropy,cast,cast_to_floatx,categorical_crossentropy,clear_session,clip,concatenate,constant,conv1d,conv2d,conv2d_transpose,conv3d,cos,count_params,ctc_batch_cost,ctc_decode,ctc_label_dense_to_sparse,dot,dropout,dtype,elu,epsilon,equal,eval,exp,expand_dims,eye,flatten,floatx,foldl,foldr,function,gather,get_session,get_uid,get_value,gradients,greater,greater_equal,hard_sigmoid,image_data_format,in_test_phase,in_top_k,in_train_phase,int_shape,is_sparse,l2_normalize,learning_phase,less,less_equal,log,manual_variable_initialization,map_fn,max,maximum,mean,min,minimum,moving_average_update,name_scope,ndim,normalize_batch_in_training,not_equal,one_hot,ones,ones_like,permute_dimensions,placeholder,pool2d,pool3d,pow,print_tensor,prod,random_binomial,random_normal,random_normal_variable,random_uniform,random_uniform_variable,relu,repeat,repeat_elements,reset_uids,reshape,resize_images,resize_volumes,reverse,rnn,round,separable_conv2d,set_epsilon,set_floatx,set_image_data_format,set_learning_phase,set_session,set_value,shape,sigmoid,sign,sin,softmax,softplus,softsign,sparse_categorical_crossentropy,spatial_2d_padding,spatial_3d_padding,sqrt,square,squeeze,stack,std,stop_gradient,sum,switch,tanh,temporal_padding,to_dense,transpose,truncated_normal,update,update_add,update_sub,var,variable,zeros,zeros_like
"""

from tensorflow.contrib.keras.python.keras.backend import elu, arange
import tensorflow as tf
import numpy as np

#######################
# create data
#######################
# from np to tf.constant
np_array = np.arange(start=-5, stop=10, step=1, dtype='float64').reshape(3,5) # safer with float
tf_constant_np = tf.constant(np_array)

# created with tf.arange
tf_arange = arange(start=-5, stop=10, step=1, dtype='float64')
tf_arange = tf.reshape(tf_arange, [3,5])

print(tf_constant_np == tf_arange)

# x is tensor or variable, (tensor: np.array, or tf.tensor are fine)
elu1 = elu(x=np_array, alpha=1.)
elu2 = elu(x=tf_constant_np, alpha=1.)
elu3 = elu(x=tf_arange, alpha=1.)


----
see source code in ipython nicely


In [21]: ?? tf.contrib.keras.backend.elu
Signature: tf.contrib.keras.backend.elu(x, alpha=1.0)
Source:
def elu(x, alpha=1.):
  """Exponential linear unit.

  Arguments:
      x: A tenor or variable to compute the activation function for.
      alpha: A scalar, slope of positive section.

  Returns:
      A tensor.
  """
  res = nn.elu(x) # tensorflow.python.ops.nn (from tensorflow ground up)
  if alpha == 1:
    return res
  else:
	# tensorflow.python.ops.array_ops
    return array_ops.where(x > 0, res, alpha * res)
File:      ~/miniconda2/envs/kur/lib/python3.6/site-packages/tensorflow/contrib/keras/python/keras/backend.py
Type:      function
