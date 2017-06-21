from tensorflow.contrib.keras.python.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1)) # make sure shape (num, width, height, channel)
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()

#Helper function to plot images by index in the validation set
#Plots is a helper function in utils.py
def plots_idx(img_dir, filenames, idx, titles=None):
    plots([image.load_img(img_dir + filenames[i]) for i in idx], titles=titles)
