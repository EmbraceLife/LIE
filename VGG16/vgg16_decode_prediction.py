import tensorflow as tf
import numpy as np

################################
## get preprocess_input function
dePred = tf.contrib.keras.applications.vgg16.decode_predictions

################################
# create fake predictions dataset based on source code
### Note
# 1. must 2D; 2. preds.shape[1] must be 1000
preds = np.random.random((8, 1000))


################################
# explore source code of tf.contrib.keras.applications.vgg16.preprocess_input
# 1. sort np.array, get the highest 5, and order their index in descending order and save into a list
# 2. get the top 5 in a list in format of [('n02808440', 'bathtub', 0.99974032567102367), ('n02480855', 'gorilla', 0.99796519466779809), ('n02917067', 'bullet_train', 0.99779524663134933), ('n03208938', 'disk_brake', 0.99765305181472064), ('n02417914', 'ibex', 0.99713595149857714)]
# 3. sort the list above in descending order on percentage values
# 4. each sample correspond to one such list, all samples saved in a bigger list
decodedPred = dePred(preds)
print(decodedPred)
