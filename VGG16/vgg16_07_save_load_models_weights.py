
##############################
# load model from h5 file

# get load_model function
from tensorflow.contrib.keras.python.keras.models import load_model

# get dir_path for models
trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"

vgg16_2class = load_model(trained_model_path+'/vgg16_2class.h5')

# del vgg16_2class  # deletes the existing model

##############################
# save model skeleton from model

# save model skeleton as JSON, but not readable directly
json_vgg16 = vgg16_2class.to_json()

# save model skeleton as YAML, but not readable directly
yaml_vgg16 = vgg16_2class.to_yaml()


##############################
# create empty model from a model skeleton in json or yaml

# model reconstruction from JSON:
from tensorflow.contrib.keras.python.keras.models import model_from_json
model_json = model_from_json(json_vgg16)

# model reconstruction from YAML
from tensorflow.contrib.keras.python.keras.models import model_from_yaml
model_yaml = model_from_yaml(yaml_vgg16)


###############################
# save only weigths from model
vgg16_2class.save_weights(trained_model_path+'/vgg_2class_weights.h5')

# load weights into empty model skeleton
model_yaml.load_weights(trained_model_path+'/vgg_2class_weights.h5')

###############################
from vgg16_finetune2 import finetune_2layers
# finetune_2layers remove current model's last layer and add two layers to it and becomes a new model
vgg16_json_new = finetune_2layers(model_json, num=2) # num=2 refer to 2 classes to predictions

# load weights of previous model to new model, only shared-name layers get weights
vgg16_json_new.load_weights(trained_model_path+'/vgg_2class_weights.h5', by_name=True)
# dr vgg16_json_new.load_weights
