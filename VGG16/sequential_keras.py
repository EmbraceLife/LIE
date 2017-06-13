###############################
## Goal: experiment on Sequential()
# which module
# attrs and methods
# inner creation process

import tensorflow as tf


model_tf_kr = tf.contrib.keras.models.Sequential()


###############################
### module and location:
# <module 'tensorflow.contrib.keras.python.keras.models' from '/Users/Natsume/miniconda2/envs/kur/lib/python3.6/site-packages/tensorflow/contrib/keras/python/keras/models.py'>

###############################
### attrs and methods: created inside Sequential class of tf_kr
# '_add_inbound_node',
#  '_assert_input_compatibility',
#  '_base_name',
#  '_compute_output_shape',
#  '_fit_loop',
#  '_gather_list_attr',
#  '_get_node_attribute_at_index',
#  '_graph',
#  '_initial_weights',
#  '_make_predict_function',
#  '_make_test_function',
#  '_make_train_function',
#  '_predict_loop',
#  '_reuse',
#  '_scope',
#  '_set_scope',
#  '_standardize_user_data',
#  '_test_loop',
#  '_trainable',
#  '_updated_config',
#  '_updates',
#  'add',
#  'add_loss',
#  'add_update',
#  'add_variable',
#  'add_weight',
#  'apply',
#  'build',
#  'built',
#  'call',
#  'compile',
#  'compute_mask',
#  'constraints',
#  'count_params',
#  'evaluate',
#  'evaluate_generator',
#  'fit',
#  'fit_generator',
#  'from_config',
#  'get_config',
#  'get_input_at',
#  'get_input_mask_at',
#  'get_input_shape_at',
#  'get_layer',
#  'get_losses_for',
#  'get_output_at',
#  'get_output_mask_at',
#  'get_output_shape_at',
#  'get_updates_for',
#  'get_weights',
#  'graph',
#  'inbound_nodes',
#  'input',
#  'input_mask',
#  'input_shape',
#  'input_spec',
#  'inputs',
#  'layers',
#  'load_weights',
#  'losses',
#  'model',
#  'name',
#  'non_trainable_variables',
#  'non_trainable_weights',
#  'outbound_nodes',
#  'output',
#  'output_mask',
#  'output_shape',
#  'outputs',
#  'pop',
#  'predict',
#  'predict_classes',
#  'predict_generator',
#  'predict_on_batch',
#  'predict_proba',
#  'regularizers',
#  'reset_states',
#  'run_internal_graph',
#  'save',
#  'save_weights',
#  'scope_name',
#  'set_weights',
#  'state_updates',
#  'stateful',
#  'summary',
#  'test_on_batch',
#  'to_json',
#  'to_yaml',
#  'train_on_batch',
#  'trainable',
#  'trainable_variables',
#  'trainable_weights',
#  'updates',
#  'uses_learning_phase',
#  'variables',
#  'weights']

###############################
### inner creation process
# def __init__(self, layers=None, name=None):
#  406         self.layers = []  # Stack of layers.
#  407         self.model = None  # Internal Model instance.
#  408         self.inputs = []  # List of input tensors
#  409         self.outputs = []  # List of length 1: the output tensor (unique).
#  410         self._trainable = True
#  411         self._initial_weights = None
#  412
#  413         # Model attributes.
#  414         self.inbound_nodes = []
#  415         self.outbound_nodes = []
#  416         self.built = False
#  417
#  418         # Set model name.
#  419         if not name:
#  420  ->       prefix = 'sequential_'
#  421           name = prefix + str(K.get_uid(prefix))
#  422         self.name = name
#  423
#  424         # The following properties are not actually used by Keras;
#  425         # they exist for compatibility with TF's variable scoping mechanism.
#  426         self._updates = []
#  427         self._scope = None
#  428         self._reuse = None
#  429         self._base_name = name
#  430         self._graph = ops.get_default_graph()
#  431
#  432         # Add to the model any layers passed to the constructor.
#  433         if layers:
#  434           for layer in layers:
#  435             self.add(layer)


import keras
model_kr = keras.models.Sequential()

################################################
### inner creation process by keras
# /Users/Natsume/miniconda2/envs/dlnd-tf-lab/lib/python3.5/site-packages/Keras-2.0.4-py3.5.egg/keras/models.py(382)
# 382  ->     def __init__(self, layers=None, name=None):
#  383             self.layers = []  # Stack of layers.
#  384             self.model = None  # Internal Model instance.
#  385             self.inputs = []  # List of input tensors
#  386             self.outputs = []  # List of length 1: the output tensor (unique).
#  387             self._trainable = True
#  388             self._initial_weights = None
#  389
#  390             # Model attributes.
#  391             self.inbound_nodes = []
#  392             self.outbound_nodes = []
#  393             self.built = False
#  394
#  395             # Set model name.
#  396             if not name:
#  397                 prefix = 'sequential_'
#  398                 name = prefix + str(K.get_uid(prefix))
#  399             self.name = name
#  400
#  401             # Add to the model any layers passed to the constructor.
#  402             if layers:
#  403                 for layer in layers:
#  404                     self.add(layer)

######################
### attrs and methods created inside Sequential class
# '_add_inbound_node',
#  '_built',
#  '_fit_loop',
#  '_flattened_layers',
#  '_gather_list_attr',
#  '_get_deduped_metrics_names',
#  '_get_node_attribute_at_index',
#  '_initial_weights',
#  '_make_predict_function',
#  '_make_test_function',
#  '_make_train_function',
#  '_predict_loop',
#  '_standardize_user_data',
#  '_test_loop',
#  '_trainable',
#  '_updated_config',
#  'add',
#  'add_loss',
#  'add_update',
#  'add_weight',
#  'assert_input_compatibility',
#  'build',
#  'built',
#  'call',
#  'compile',
#  'compute_mask',
#  'compute_output_shape',
#  'constraints',
#  'count_params',
#  'evaluate',
#  'evaluate_generator',
#  'fit',
#  'fit_generator',
#  'from_config',
#  'get_config',
#  'get_input_at',
#  'get_input_mask_at',
#  'get_input_shape_at',
#  'get_layer',
#  'get_losses_for',
#  'get_output_at',
#  'get_output_mask_at',
#  'get_output_shape_at',
#  'get_updates_for',
#  'get_weights',
#  'inbound_nodes',
#  'input',
#  'input_mask',
#  'input_shape',
#  'input_spec',
#  'inputs',
#  'layers',
#  'legacy_from_config',
#  'legacy_get_config',
#  'load_weights',
#  'losses',
#  'model',
#  'name',
#  'non_trainable_weights',
#  'outbound_nodes',
#  'output',
#  'output_mask',
#  'output_shape',
#  'outputs',
#  'pop',
#  'predict',
#  'predict_classes',
#  'predict_generator',
#  'predict_on_batch',
#  'predict_proba',
#  'regularizers',
#  'reset_states',
#  'run_internal_graph',
#  'save',
#  'save_weights',
#  'set_weights',
#  'state_updates',
#  'stateful',
#  'summary',
#  'test_on_batch',
#  'to_json',
#  'to_yaml',
#  'train_on_batch',
#  'trainable',
#  'trainable_weights',
#  'updates',
#  'uses_learning_phase',
#  'weights']
