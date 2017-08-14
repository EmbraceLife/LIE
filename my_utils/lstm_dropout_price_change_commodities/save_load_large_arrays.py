"""
save_load_large_arrays
- can I save a list of large arrays?
"""


#############################################
# save large arrays using bcolz

import bcolz
# create a folder and save data inside
def bz_save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def bz_load_array(fname):
    return bcolz.open(fname)[:]

# save_array(trained_model_path+"/preds_bc", preds)
# preds_bc = load_array(trained_model_path+"/preds_bc")



#############################################
# # save arrays using numpy
#
# # save preds for decoding
# import numpy as np
# def np_save(dir_path, large_array):
#     np.save(dir_path, large_array)
#
# # load preds.npy
# def np_load(npy_file_path):
#     return np.load(npy_file_path)


#############################################
# # save arrays or objects in pickle
# import pickle
#
# def pk_save(dir_path, large_array):
#     with open(dir_path+".pickle", "wb") as f:
#         pickle.dump(large_array, f)
#
# def pk_load(dir_path):
#     with open(dir_path+".pickle", "rb") as f:
#         large_array = pickle.load(f)
#     return large_array


# ######################
# # save and load with torch.save, torch.load
# import torch
# torch.save(train_img_array, trained_model_path+"/train_img_array_torch")
# train_img_array_torch = torch.load(trained_model_path+"/train_img_array_torch")


#############################################
# # save and load with kur.idx
# from kur.utils import idx
#
# def idx_save(dir_path, large_array):
#     idx.save(dir_path, large_array)
#
# def idx_load(dir_path):
#     return idx.load(dir_path)
