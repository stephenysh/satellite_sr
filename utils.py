import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *

import scipy
from matplotlib.pyplot import imread
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    return (imread(path + file_name)*255).astype('uint8')

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx: idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d image from %s' % (len(imgs), path))
    return imgs