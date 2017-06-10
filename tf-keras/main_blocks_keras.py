import tensorflow as tf

########################
# major blocks inside keras
# dr tf.contrib.keras
"""
(Pdb++) dr tf.contrib.keras
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__', # source loader
 '__name__',
 '__package__',
 '__path__',
 '__spec__',     # module spec
 'activations',  # Keras built-in activation functions
 'api',          # contains everything
 'applications', # Keras Applications are canned architectures
 				   with pre-trained weights.
 'backend',		 # Keras backend API
 'callbacks',	 # Keras callback classes
 'constraints',  # Keras built-in constraints functions
 'datasets',	 # Keras built-in datasets
 'initializers', # Keras built-in initializers
 'layers',		 # Keras layers API
 'losses',		 # Keras built-in loss functions
 'metrics',		 # Keras built-in metrics functions
 'models',		 # Keras models API
 'optimizers',	 # Keras built-in optimizers
 'preprocessing',# Keras data preprocessing utils
 'regularizers', # Keras built-in regularizers
 'utils',		 # Keras utils
 'wrappers']	 # Wrappers for Keras models, providing
 				 compatibility with other frameworks
"""

#############################
# activations block
# dr tf.contrib.keras.activations
"""
(Pdb++) dr tf.contrib.keras.activations
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'deserialize',
 'elu',
 'get',
 'hard_sigmoid',
 'linear',
 'relu',
 'serialize',
 'sigmoid',
 'softmax',
 'softplus',
 'softsign',
 'tanh']
"""

act_func = tf.contrib.keras.activations.get('relu')


##########################
# api block

"""
(Pdb++) dr tf.contrib.keras.api
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'keras']

(Pdb++) dr tf.contrib.keras.api.keras
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'activations',
 'applications',
 'backend',
 'callbacks',
 'constraints',
 'datasets',
 'initializers',
 'layers',
 'losses',
 'metrics',
 'models',
 'optimizers',
 'preprocessing',
 'regularizers',
 'utils',
 'wrappers']
 """

 ##########################
 # applications block

"""
 (Pdb++) dr tf.contrib.keras.applications
['InceptionV3',
 'ResNet50',
 'VGG16',
 'VGG19',
 'Xception',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'inception_v3',
 'resnet50',
 'vgg16',
 'vgg19',
 'xception']
"""

###########################
# backend block: backend

"""
(Pdb++) dr tf.contrib.keras.backend
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'abs',
 'all',
 'any',
 'arange',
 'argmax',
 'argmin',
 'backend',
 'batch_dot',
 'batch_flatten',
 'batch_get_value',
 'batch_normalization',
 'batch_set_value',
 'bias_add',
 'binary_crossentropy',
 'cast',
 'cast_to_floatx',
 'categorical_crossentropy',
 'clear_session',
 'clip',
 'concatenate',
 'constant',
 'conv1d',
 'conv2d',
 'conv2d_transpose',
 'conv3d',
 'cos',
 'count_params',
 'ctc_batch_cost',
 'ctc_decode',
 'ctc_label_dense_to_sparse',
 'dot',
 'dropout',
 'dtype',
 'elu',
 'epsilon',
 'equal',
 'eval',
 'exp',
 'expand_dims',
 'eye',
 'flatten',
 'floatx',
 'foldl',
 'foldr',
 'function',
 'gather',
 'get_session',
 'get_uid',
 'get_value',
 'gradients',
 'greater',
 'greater_equal',
 'hard_sigmoid',
 'image_data_format',
 'in_test_phase',
 'in_top_k',
 'in_train_phase',
 'int_shape',
 'is_sparse',
 'l2_normalize',
 'learning_phase',
 'less',
 'less_equal',
 'log',
 'manual_variable_initialization',
 'map_fn',
 'max',
 'maximum',
 'mean',
 'min',
 'minimum',
 'moving_average_update',
 'name_scope',
 'ndim',
 'normalize_batch_in_training',
 'not_equal',
 'one_hot',
 'ones',
 'ones_like',
 'permute_dimensions',
 'placeholder',
 'pool2d',
 'pool3d',
 'pow',
 'print_tensor',
 'prod',
 'random_binomial',
 'random_normal',
 'random_normal_variable',
 'random_uniform',
 'random_uniform_variable',
 'relu',
 'repeat',
 'repeat_elements',
 'reset_uids',
 'reshape',
 'resize_images',
 'resize_volumes',
 'reverse',
 'rnn',
 'round',
 'separable_conv2d',
 'set_epsilon',
 'set_floatx',
 'set_image_data_format',
 'set_learning_phase',
 'set_session',
 'set_value',
 'shape',
 'sigmoid',
 'sign',
 'sin',
 'softmax',
 'softplus',
 'softsign',
 'sparse_categorical_crossentropy',
 'spatial_2d_padding',
 'spatial_3d_padding',
 'sqrt',
 'square',
 'squeeze',
 'stack',
 'std',
 'stop_gradient',
 'sum',
 'switch',
 'tanh',
 'temporal_padding',
 'to_dense',
 'transpose',
 'truncated_normal',
 'update',
 'update_add',
 'update_sub',
 'var',
 'variable',
 'zeros',
 'zeros_like']
"""



############################
# callbacks blocks
# create logs and tensorboard

"""
(Pdb++) dr tf.contrib.keras.callbacks
['BaseLogger',
 'CSVLogger',
 'Callback',
 'EarlyStopping',
 'History',
 'LambdaCallback',
 'LearningRateScheduler',
 'ModelCheckpoint',
 'ProgbarLogger',
 'ReduceLROnPlateau',
 'RemoteMonitor',
 'TensorBoard',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__']
"""

###########################
# constraints blocks
# to constrain the values
"""
(Pdb++) dr tf.contrib.keras.constraints
['Constraint',
 'MaxNorm',
 'MinMaxNorm',
 'NonNeg',
 'UnitNorm',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'deserialize',
 'get',
 'max_norm',
 'min_max_norm',
 'non_neg',
 'serialize',
 'unit_norm']
"""

###########################
# datasets blocks
"""
(Pdb++) dr tf.contrib.keras.datasets
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'boston_housing',
 'cifar10',
 'cifar100',
 'imdb',
 'mnist',
 'reuters']
"""

#####################
# initializers block

"""
(Pdb++) dr tf.contrib.keras.initializers
['Constant',
 'Identity',
 'Initializer',
 'Ones',
 'Orthogonal',
 'RandomNormal',
 'RandomUniform',
 'TruncatedNormal',
 'VarianceScaling',
 'Zeros',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'deserialize',
 'get',
 'glorot_normal',
 'glorot_uniform',
 'he_normal',
 'he_uniform',
 'lecun_uniform',
 'serialize']
"""


######################
# layers blocks

"""
(Pdb++) dr tf.contrib.keras.layers
['Activation',
 'ActivityRegularization',
 'Add',
 'Average',
 'AveragePooling1D',
 'AveragePooling2D',
 'AveragePooling3D',
 'AvgPool1D',
 'AvgPool2D',
 'AvgPool3D',
 'BatchNormalization',
 'Bidirectional',
 'Concatenate',
 'Conv1D',
 'Conv2D',
 'Conv2DTranspose',
 'Conv3D',
 'ConvLSTM2D',
 'Convolution1D',
 'Convolution2D',
 'Convolution2DTranspose',
 'Convolution3D',
 'Cropping1D',
 'Cropping2D',
 'Cropping3D',
 'Dense',
 'Dot',
 'Dropout',
 'ELU',
 'Embedding',
 'Flatten',
 'GRU',
 'GaussianDropout',
 'GaussianNoise',
 'GlobalAveragePooling1D',
 'GlobalAveragePooling2D',
 'GlobalAveragePooling3D',
 'GlobalAvgPool1D',
 'GlobalAvgPool2D',
 'GlobalAvgPool3D',
 'GlobalMaxPool1D',
 'GlobalMaxPool2D',
 'GlobalMaxPool3D',
 'GlobalMaxPooling1D',
 'GlobalMaxPooling2D',
 'GlobalMaxPooling3D',
 'Input',
 'InputLayer',
 'InputSpec',
 'LSTM',
 'Lambda',
 'Layer',
 'LeakyReLU',
 'LocallyConnected1D',
 'LocallyConnected2D',
 'Masking',
 'MaxPool1D',
 'MaxPool2D',
 'MaxPool3D',
 'MaxPooling1D',
 'MaxPooling2D',
 'MaxPooling3D',
 'Maximum',
 'Multiply',
 'PReLU',
 'Permute',
 'RepeatVector',
 'Reshape',
 'SeparableConv2D',
 'SeparableConvolution2D',
 'SimpleRNN',
 'SpatialDropout1D',
 'SpatialDropout2D',
 'SpatialDropout3D',
 'ThresholdedReLU',
 'TimeDistributed',
 'UpSampling1D',
 'UpSampling2D',
 'UpSampling3D',
 'Wrapper',
 'ZeroPadding1D',
 'ZeroPadding2D',
 'ZeroPadding3D',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'add',
 'average',
 'concatenate',
 'dot',
 'maximum',
 'multiply']
"""
# sources tf.contrib.keras.layers.add


###################
# losses block

"""
(Pdb++) dr tf.contrib.keras.losses
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'binary_crossentropy',
 'categorical_crossentropy',
 'cosine_proximity',
 'deserialize',
 'get',
 'hinge',
 'kullback_leibler_divergence',
 'mean_absolute_error',
 'mean_absolute_percentage_error',
 'mean_squared_error',
 'mean_squared_logarithmic_error',
 'poisson',
 'serialize',
 'sparse_categorical_crossentropy',
 'squared_hinge']
"""

#######################
# metrics blocks

"""
(Pdb++) dr tf.contrib.keras.metrics
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'binary_accuracy',
 'binary_crossentropy',
 'categorical_accuracy',
 'categorical_crossentropy',
 'cosine_proximity',
 'deserialize',
 'get',
 'hinge',
 'kullback_leibler_divergence',
 'mean_absolute_error',
 'mean_absolute_percentage_error',
 'mean_squared_error',
 'mean_squared_logarithmic_error',
 'poisson',
 'serialize',
 'sparse_categorical_crossentropy',
 'squared_hinge',
 'top_k_categorical_accuracy']
"""

######################
# models block

"""
(Pdb++) dr tf.contrib.keras.models
['Model',
 'Sequential',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'load_model',
 'model_from_config',
 'model_from_json',
 'model_from_yaml',
 'save_model']
"""

"""
(Pdb++) dr tf.contrib.keras.models.Sequential
['__call__',
 '__class__',
 '__deepcopy__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 '_add_inbound_node',
 '_assert_input_compatibility',
 '_compute_output_shape',
 '_fit_loop',
 '_gather_list_attr',
 '_get_node_attribute_at_index',
 '_make_predict_function',
 '_make_test_function',
 '_make_train_function',
 '_predict_loop',
 '_set_scope',
 '_standardize_user_data',
 '_test_loop',
 '_updated_config',
 'add',
 'add_loss',
 'add_update',
 'add_variable',
 'add_weight',
 'apply',
 'build',
 'call',
 'compile',
 'compute_mask',
 'constraints',
 'count_params',
 'evaluate',
 'evaluate_generator',
 'fit',
 'fit_generator',
 'from_config',
 'get_config',
 'get_input_at',
 'get_input_mask_at',
 'get_input_shape_at',
 'get_layer',
 'get_losses_for',
 'get_output_at',
 'get_output_mask_at',
 'get_output_shape_at',
 'get_updates_for',
 'get_weights',
 'graph',
 'input',
 'input_mask',
 'input_shape',
 'input_spec',
 'load_weights',
 'losses',
 'non_trainable_variables',
 'non_trainable_weights',
 'output',
 'output_mask',
 'output_shape',
 'pop',
 'predict',
 'predict_classes',
 'predict_generator',
 'predict_on_batch',
 'predict_proba',
 'regularizers',
 'reset_states',
 'run_internal_graph',
 'save',
 'save_weights',
 'scope_name',
 'set_weights',
 'state_updates',
 'stateful',
 'summary',
 'test_on_batch',
 'to_json',
 'to_yaml',
 'train_on_batch',
 'trainable',
 'trainable_variables',
 'trainable_weights',
 'updates',
 'uses_learning_phase',
 'variables',
 'weights']
"""

"""
(Pdb++) dr tf.contrib.keras.models.Model
['__call__',
 '__class__',
 '__deepcopy__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 '_add_inbound_node',
 '_assert_input_compatibility',
 '_compute_output_shape',
 '_fit_loop',
 '_get_node_attribute_at_index',
 '_make_predict_function',
 '_make_test_function',
 '_make_train_function',
 '_predict_loop',
 '_set_scope',
 '_standardize_user_data',
 '_test_loop',
 '_updated_config',
 'add_loss',
 'add_update',
 'add_variable',
 'add_weight',
 'apply',
 'build',
 'call',
 'compile',
 'compute_mask',
 'constraints',
 'count_params',
 'evaluate',
 'evaluate_generator',
 'fit',
 'fit_generator',
 'from_config',
 'get_config',
 'get_input_at',
 'get_input_mask_at',
 'get_input_shape_at',
 'get_layer',
 'get_losses_for',
 'get_output_at',
 'get_output_mask_at',
 'get_output_shape_at',
 'get_updates_for',
 'get_weights',
 'graph',
 'input',
 'input_mask',
 'input_shape',
 'input_spec',
 'load_weights',
 'losses',
 'non_trainable_variables',
 'non_trainable_weights',
 'output',
 'output_mask',
 'output_shape',
 'predict',
 'predict_generator',
 'predict_on_batch',
 'reset_states',
 'run_internal_graph',
 'save',
 'save_weights',
 'scope_name',
 'set_weights',
 'state_updates',
 'stateful',
 'summary',
 'test_on_batch',
 'to_json',
 'to_yaml',
 'train_on_batch',
 'trainable_variables',
 'trainable_weights',
 'updates',
 'uses_learning_phase',
 'variables',
 'weights']

"""

########################
# optimizers block

"""
Pdb++) dr tf.contrib.keras.optimizers
['Adadelta',
 'Adagrad',
 'Adam',
 'Adamax',
 'Nadam',
 'Optimizer',
 'RMSprop',
 'SGD',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'deserialize',
 'get',
 'serialize']
"""


############################
# preprocessing block

"""
(Pdb++) dr tf.contrib.keras.preprocessing
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'image',
 'sequence',
 'text']
"""


##############################
# regularizers block
"""
(Pdb++) dr tf.contrib.keras.regularizers
['L1L2',
 'Regularizer',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'deserialize',
 'get',
 'l1',
 'l1_l2',
 'l2',
 'serialize']
"""


################################
# utils block

"""
(Pdb++) dr tf.contrib.keras.utils
['CustomObjectScope',
 'HDF5Matrix',
 'Progbar',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'convert_all_kernels_in_model',
 'custom_object_scope',
 'deserialize_keras_object',
 'get_custom_objects',
 'get_file',
 'normalize',
 'plot_model',
 'serialize_keras_object',
 'to_categorical']
"""

###############################
# wrappers block

"""
(Pdb++) dr tf.contrib.keras.wrappers
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 'scikit_learn']
"""
