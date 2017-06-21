"""
1. from a trained model to save model and weights
2. from saved model file and weight file, to load a new trained model
3. extract a json or yaml representation of a model
4. create an untrained model from json or yaml representation
5. load weights into an untrained model
"""

# get load_model function
from tensorflow.contrib.keras.python.keras.models import load_model, save_model, model_from_yaml, model_from_json

# the following 5 methods can be directly used by Model object
# from tensorflow.contrib.keras.python.keras.topology.Container import save, save_weights, load_weights, to_json, to_yaml

# access a trained model (it will be traied for the first time and then saved later below)
from train_model_01_train_with_fit_generator_vgg16 import vgg16_ft_trained
# a model object as a subclass of tf.keras.Container has 3 class methods
# 1. model.save or Container.save to save trained model
# 1. trained_model_loaded = load_model() imported from tf.keras.models
# 2. model.save_weights or Container.save_weights to save only trained weights
# 3. model.load_weights or Container.load_weights to load trained weights to an untrained model

trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"


##############################
# save a trained model
vgg16_ft_trained.save(trained_model_path+'/vgg16_ft_trained_model.h5')

##############################
# load model
vgg16_ft_trained_new = load_model(trained_model_path+'/vgg16_ft_trained_model.h5')

##############################
# save trained weights from a trained model
vgg16_ft_trained.save_weights(trained_model_path+'/vgg16_ft_trained_weights.h5')


##############################
# save model skeleton from model

# save model skeleton as JSON, but not readable directly
json_vgg16 = vgg16_ft_trained.to_json()

# save model skeleton as YAML, but not readable directly
yaml_vgg16 = vgg16_ft_trained.to_yaml()


##############################
# create empty model from a model skeleton in json or yaml
# model reconstruction from JSON:
model_json = model_from_json(json_vgg16)

# model reconstruction from YAML
model_yaml = model_from_yaml(yaml_vgg16)

##############################
# feed an untrained model with weights
model_yaml.load_weights(trained_model_path+'/vgg16_ft_trained_weights.h5')


###############################
# three models share the same graph and same collections of weights and biases
vgg16_ft_trained_new.graph._collections == vgg16_ft_trained.graph._collections == model_yaml.graph._collections
