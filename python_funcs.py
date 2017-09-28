import cv2
import glob
import pickle

from ipywidgets import widgets
from ipywidgets import interact

from skimage.feature import hog

from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from moviepy.editor import VideoFileClip
import numpy as np
from sys import platform


import os
# %matplotlib inline // Jupyter Notebooks only

print('Done importing everything.  System ready to rip!')


#### Next cell ####

def process_base_image_dir(basedir):
    # Different folders are different sources for images, GTI, KITTI, etc
    img_types = os.listdir(basedir)
    if platform == 'darwin':  # Mac OSX
        print('OSX environemnt, trying to remove DS_store')
        if '.DS_Store' in img_types: img_types.remove('.DS_Store')

    print('Found types: {}'.format(img_types))
    return img_types

def process_image_types(image_types, file_name, type_name):
    images = []
    for imtype in image_types:
        images.extend(glob.glob("{}/{}/*".format(basedir, imtype)))
    print('Number of {} Images found: {}'.format(type_name, len(images)))
    with open(file_name, 'w') as f:
        for fn in images:
            f.write('{}{}'.format(fn, os.linesep))

    return images

basedir = './data/vehicles'
image_types = process_base_image_dir(basedir)
cars = process_image_types(image_types, 'cars.txt', 'Vehicle')

basedir = './data/non-vehicles'
image_types = os.listdir(basedir)
notcars = process_image_types(image_types, 'notcars.txt', 'Non-Vehicle')


#### Next cell ####



