import cv2
import glob
import pickle

import matplotlib
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
#%matplotlib inline #// Jupyter Notebooks only
from sklearn.preprocessing import StandardScaler

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

#### Next Cell ###
# Utility Function

def correct_for_colorspace_or_copy(image, color_space):
    feature_image = None
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)

    return feature_image


#### Next cell ####
# Feature-specific functions
def base_hog(should_visualize, img, orient, pix_per_cell, cell_per_block, feature_vec):
    return hog(
            img,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=False,
            visualise=should_visualize,
            feature_vector=feature_vec
        )

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    return base_hog(vis, img, orient, pix_per_cell, cell_per_block, feature_vec)

# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_image = correct_for_colorspace_or_copy(img, color_space=color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32): #, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel_1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel_2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel_3_hist = np.histogram(img[:, :, 2], bins=nbins)

    # Generating bin centers
    #bin_edges = channel_1_hist[1]
    #bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

    hist_features = np.concatenate((channel_1_hist[0], channel_2_hist[0], channel_3_hist[0]))

    return hist_features
    #return channel_1_hist, channel_2_hist, channel_3_hist, bin_centers, hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(
        imgs,
        color_space='RGB',
        spatial_size=(32, 32),
        hist_bins=32,
        # hist_range=(0, 256),
        orient=9,
        pix_per_cell=9,
        cell_per_block=2,
        hog_channel=0, # or ALL
        spatial_feat=False,
        hist_feat=False,
        hog_feat=True
):
    features = []
    for file in imgs:
        file_features = []
        image = mpimg.imread(file)
        feature_image = correct_for_colorspace_or_copy(image, color_space)

        ### NOTE::: Extracting features need to be done in the same order as here
        ### NOTE::: Extracting features need to be done in the same order as here
        ### NOTE::: Extracting features need to be done in the same order as here
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins) #, bins_range=hist_range)
            file_features.append(hist_features)
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(
                        get_hog_features(
                            feature_image[:,:,channel],
                            orient=orient,
                            pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            vis=False,
                            feature_vec=True
                        )
                    )
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(
                    feature_image[:, :, hog_channel],
                    orient=orient,
                    pix_per_cell=pix_per_cell,
                    cell_per_block=cell_per_block,
                    vis=False,
                    feature_vec=True
                )

            file_features.append(hog_features)

        #features.append(np.concatenate((spatial_features, hist_features)))
        features.append(np.concatenate(file_features))

    return features


#### Next cell ####
# Testing stuff out

if platform != 'darwin':  # Mac OSX
    print('Only meant for running from command line!')
    #return
else:
    test_images = glob.glob("./test_images/*.png")

    features = extract_features(
        test_images,
        color_space='RGB',
        spatial_size=(32, 32),
        hist_bins=32
        #, hist_range=(0, 256)
    )

    if len(features) == 0:
        print('Your function only returns empty feature vectors...')
    else:
        # Create an array stack of feature vectors
        X = np.vstack(features).astype(np.float64) #, notcar_features
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        features_ind = np.random.randint(0, len(features))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(test_images[features_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[features_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[features_ind])
        plt.title('Normalized Features')
        fig.tight_layout()

        plt.savefig('./output_images/test_feature_hog_only.png')
        #plt.show()

#### Next cell ###
