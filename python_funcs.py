import csv

import cv2
import glob
import pickle

import matplotlib
import time
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
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

from scipy.ndimage.measurements import label

# %matplotlib inline #// Jupyter Notebooks only

print('Done importing everything.  System ready to rip!')


#### Next cell ####
# Initial importing of data.

def process_base_image_dir(basedir):
    if not os.path.isdir(basedir):
        raise BaseException(basedir, 'folder not found! Please see ./data/README.md for downloading links and expected structure of data')


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
    #with open(file_name, 'w') as f:
    #    for fn in images:
     #       f.write('{}{}'.format(fn, os.linesep))

    return images


basedir = './data/vehicles'
image_types = process_base_image_dir(basedir)
cars = process_image_types(image_types, 'cars.txt', 'Vehicle')
#print('cars sample shape is: {}'.format(cars[0].shape))

basedir = './data/non-vehicles'
image_types = os.listdir(basedir)
notcars = process_image_types(image_types, 'notcars.txt', 'Non-Vehicle')
#print('noncars sample shape is: {}'.format(notcars[0].shape))

basedir = './data/object-detection-crowdai'

def load_crowd_ai():
    if not os.path.isdir(basedir):
        raise BaseException(basedir, 'folder not found! Please see ./data/README.md for downloading links and expected structure of data')

    IMG_IDX = 4
    LABEL_IDX = 5

    count = 1
    with open('{}/labels.csv'.format(basedir)) as csvfile:
        before_cars_len = len(cars)
        before_notcars_len = len(notcars)

        reader = csv.reader(csvfile)
        next(reader, None) # skip header

        cur_frame = None
        cur_frame_img = None
        for line in reader:
            if line[IMG_IDX] != cur_frame:
                cur_frame = line[IMG_IDX]
                input_file = '{}/{}'.format(basedir, cur_frame)
                cur_frame_img = cv2.imread(input_file)#.transpose(0,1)
                #print(cur_frame_img.shape, 'loaded image')

            xmin = int(line[0])
            ymin = int(line[1])
            xmax = int(line[2])
            ymax = int(line[3])
            #print((xmin, xmax), (ymin, ymax))
            cropped = cur_frame_img[ymin:ymax, xmin:xmax]
            #print('Cropped size (pre-resizing): {}'.format(cropped.shape))
            #if not all(cropped.shape):
            #    continue
            cropped = cv2.resize(cropped, (64, 64))

            if line[LABEL_IDX] == 'Car':
                outfile_name = '{}/{}/img_{}.png'.format(basedir, 'cars', count)
                cars.append(outfile_name)
            else:
                outfile_name = '{}/{}/img_{}.png'.format(basedir, 'notcars', count)
                notcars.append(outfile_name)

            if not os.path.isfile(outfile_name):
                cv2.imwrite(outfile_name, cropped)

            #plt.imshow(cropped)
            #plt.axis('off')
            #plt.savefig(outfile_name)

            count += 1

            if count > 5000:
                print('Added 5,000 images, is that enough?!')
                break

        print('Added {} cars from Udacity CrowdAI annotated set'.format(len(cars) - before_cars_len))
        print('Added {} notcars'.format(len(notcars) - before_notcars_len))

load_crowd_ai()

del basedir, image_types

def from_test_images(test_images):
    cars = []
    notcars = []

    for file_name in test_images:
        if 'extra' in file_name \
                or ('image' in file_name.split('/')[-1] and '00' not in file_name):
            notcars.append(file_name)
        else:
            cars.append(file_name)

    return cars, notcars

def from_data_set(num_samples=None):
    if (num_samples != None):
        random_car_idxs = np.random.randint(0, len(cars), num_samples)
        random_notcar_idxs = np.random.randint(0, len(notcars), num_samples)
        sample_cars = np.array(cars)[random_car_idxs]
        sample_notcars = np.array(notcars)[random_notcar_idxs]
        return sample_cars, sample_notcars
    else:
        return cars, notcars


print('Printed above size of test sets, also imported cars and notcars and very useful helper functions to load')


#### Next Cell ###
# Utility Function

ALL_HOG_CHANNELS = 'ALL'

SHOULD_TRAIN_CLASSIFIER=True # False will load saved model instead of training
SHOULD_RECOMPUTE_FEATURES=True # False will load saved model instead of extracting features from training set

X_SCALER_FILE = 'X_scaler_pickle.p'
SVC_PICKLE_FILE = 'svc_pickle.p'
FEATURE_PICKLE_FILE = 'features_pickle.p'

## THIS HAS ALL THE PARAMS TUNED
def common_params(
        func_to_apply,
        deboog_name,
        color_space = 'YCrCb',  # Also can be RGB, HSV, LUV, HLS, YUV, YCrCb
        spatial_size = (14, 14), # results in 3,072 with 16x16, 12,288 with 64x64 and 192 for 8x8;, or x * y * 3
        #  ^^ with minimal training, seems accuracy drops...
        hist_bins = 32, # 16&32 results in 96, 64 in length 192
        orient = 9,
        pix_per_cell = 8, # 4 results in 24,300 length feature
        cell_per_block = 2,
        hog_channel = ALL_HOG_CHANNELS,
        spatial_feat=True,
        hist_feat=True,
        hog_feat=True,  # results in 5,292 with 8x2x9 (cell/block,pix/cell,orient)
        ):

    print('Using:', orient, 'orientations',
          pix_per_cell, 'pixels per cell and',
          cell_per_block, 'cells per block')

    def call_with_input(input):
        print('{}...'.format(deboog_name))
        return func_to_apply(
            *input,
            color_space=color_space,
            spatial_size=spatial_size,
            hist_bins=hist_bins,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            hog_channel=hog_channel,
            spatial_feat=spatial_feat,
            hist_feat=hist_feat,
            hog_feat=hog_feat
            )

    return call_with_input

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

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

def load_classifier():
    print('Loading SVC and X_Scaler')
    with open(SVC_PICKLE_FILE, "rb") as file:
        svc = pickle.load(file)
    with open(X_SCALER_FILE, "rb") as file:
        X_scaler = pickle.load(file)

    return X_scaler, svc

OUTPUT_DIR = './output_images/{}'

if platform == 'darwin':
    print('Presuming to be on a mac, everything will be saved to file.')
else:
    print('Presuming to be in Jupyter Notebook, calling show()')

print('Loaded all utility functions and some constants')

#### Next Cell ####
#####  Functions for drawing and other image manipulation #####

def show_or_save(output_file_name='did_not_supply_file_name.png'):
    if platform == 'darwin':
        #print('Presuming to be on a mac, saving to file')
        plt.savefig(OUTPUT_DIR.format(output_file_name))
    else:
        #print('Presuming to be in Jupyter Notebook, calling show()')
        plt.show()

    plt.figure()

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img

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
        feature_vector=feature_vec,
        block_norm='L1'
        )


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, viz=False, feature_vec=True):
    return base_hog(viz, img, orient, pix_per_cell, cell_per_block, feature_vec)


# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space, size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_image = correct_for_colorspace_or_copy(img, color_space=color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel_1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel_2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel_3_hist = np.histogram(img[:, :, 2], bins=nbins)

    # Generating bin centers
    # bin_edges = channel_1_hist[1]
    # bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

    hist_features = np.concatenate((channel_1_hist[0], channel_2_hist[0], channel_3_hist[0]))

    return hist_features
    # return channel_1_hist, channel_2_hist, channel_3_hist, bin_centers, hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(
        imgs,
        color_space,
        spatial_size=(24, 24), # next try 32
        hist_bins=24,
        orient=9,
        pix_per_cell=8,
        cell_per_block=2,
        hog_channel=ALL_HOG_CHANNELS,
        spatial_feat=True,
        hist_feat=True,
        hog_feat=True
        ):
    features = []
    for file in imgs:
        image = mpimg.imread(file)
        img_features = single_img_features(
            image,
            color_space,
            hist_bins,
            hist_feat,
            hog_channel,
            hog_feat,
            cell_per_block,
            orient,
            pix_per_cell,
            spatial_feat,
            spatial_size
            )

        if len(img_features) != 0: features.append(img_features)

    return features


def hog_params(feature_image, orient, pix_per_cell, cell_per_block, feature_vec):
    def to_call(channel, viz):
        return get_hog_features(
            feature_image[:, :, channel],
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            viz=viz,
            feature_vec=True
            )

    return to_call


def single_img_features(
        image,
        # CLEANME: Instead, pass in list of features that return what to append, and then no need for all these params...
        color_space,
        hist_bins,
        hist_feat,
        hog_channel,
        hog_feat,
        cell_per_block,
        orient,
        pix_per_cell,
        spatial_feat,
        spatial_size
        #viz = False,  # CLEANME: Remove this, muddied up API and not needed here.
        ):
    viz = False # turn true for use with visualize_hog_features
    img_features = []
    hog_images = []
    feature_image = correct_for_colorspace_or_copy(image, color_space)
    ### NOTE::: Extracting features need to be done in the same order as here
    ### NOTE::: Extracting features need to be done in the same order as here
    ### NOTE::: Extracting features need to be done in the same order as here
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, color_space=color_space, size=spatial_size)
        img_features.append(spatial_features)
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    if hog_feat == True:
        hog_func = hog_params(feature_image, orient, pix_per_cell, cell_per_block, feature_vec=True)
        if hog_channel == ALL_HOG_CHANNELS:
            hog_features = []
            for channel in range(feature_image.shape[2]):
                #hoggy_feature = None
                if viz:
                    hoggy_feature, hog_image = hog_func(channel, viz)
                    hog_images.append(hog_image)
                else:
                    hoggy_feature = hog_func(channel, viz)

                hog_features.append(hoggy_feature)

            hog_features = np.ravel(hog_features)
            if viz:
                hog_images = np.vstack(hog_images)
        else:
            if viz:
                hog_features, hog_image = hog_func(hog_channel, viz)
                hog_images.append(hog_image)
            else:
                hog_features = hog_func(hog_channel, viz)

        img_features.append(hog_features)

    # features.append(np.concatenate((spatial_features, hist_features)))
    if (len(img_features) == 0):
        print('No features produced, did you turn them all off or some other exception?')
        return []

    result = np.concatenate(img_features)
    if viz:
        return result, hog_images
    else:
        return result

print('Loaded Feature Extract Helper Functions!')


#### Next cell ####

# Sliding window and such!

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(
        img,
        x_start_stop=[None, None],
        y_start_stop=[None, None],
        xy_window=(64, 64),
        xy_overlap=(0.5, 0.5)
        ):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(
        img,
        ystart,
        ystop,
        scale,
        svc,
        X_scaler,
        orient,
        pix_per_cell,
        cell_per_block,
        spatial_size,
        hist_bins,
        color_space
        ):

    img_boxes = []
    #draw_img = np.copy(img)
    count = 0
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    on_positive_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, color_space=color_space, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                #cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                #              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                # Calculate window position
                startx = xbox_left #xs * nx_pix_per_step + x_start_stop[0]
                endx = xbox_left + win_draw
                starty = ytop_draw + ystart
                endy = ytop_draw + win_draw + ystart
                # Append window position to list

                on_positive_windows.append(((startx, starty), (endx, endy)))
                img_boxes.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

    return on_positive_windows, heatmap

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(
        img,
        windows,
        clf,
        scaler,
        color_space,
        spatial_size,
        hist_bins,
        orient,
        pix_per_cell,
        cell_per_block,
        hog_channel,
        spatial_feat,
        hist_feat,
        hog_feat
        ):

    on_positive_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(
            image=test_img,
            color_space=color_space,
            spatial_size=spatial_size,
            hist_bins=hist_bins,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            hog_channel=hog_channel,
            spatial_feat=spatial_feat,
            hist_feat=hist_feat,
            hog_feat=hog_feat
            )
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_positive_windows.append(window)
    return on_positive_windows


print('Loaded up sliding window functions.  Watch out for that banana peel!')

##### Next Cell ####
# The training and the pipeline, bringing it all together in the big kahuna!

def train_classifier(
        extract,
        C=500.0, # Yes...very large C-- but we are doing hard negative mining, works quite effectively >:-D
        ):

    if SHOULD_RECOMPUTE_FEATURES:
        # Read in cars and notcars
        t1 = time.time()
        cars, notcars = from_data_set() #num_samples=1500)
        # from_test_images(test_images)

        car_features = extract([cars])
        notcar_features = extract([notcars])
        print('Saving loaded features')
        with open('car_features.pickle', 'wb') as file:
            pickle.dump(car_features, file)
        with open('notcar_features.pickle', 'wb') as file:
            pickle.dump(notcar_features, file)

        t2 = time.time()
        print(round(t2 - t1, 2), 'Seconds to extract all the features from all the things...')
    else :
        print('Loading features from pickles')
        with open('car_features.pickle', "rb") as file:
             car_features = pickle.load(file)
        with open('notcar_features.pickle', "rb") as file:
            notcar_features = pickle.load(file)

    # else:
    #     extract = common_params(extract_features, 'Extracting features')
    #     features = extract([test_images])
    #     with open(FEATURE_PICKLE_FILE, "wb") as file:
    #         pickle.dump(features, file)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Feature vector length:', len(X_train[0]))

    print('Training Classifier with C: {}...'.format(C))

    # Use a linear SVC
    svc = LinearSVC(C=C)
    ##svc = XGBClassifier() ### TODO: WANT TO TRY...but need to install...no time
    #svc = RandomForestClassifier(n_estimators=100)

    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), '...Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    print('Saving SVC and X_Scaler')
    with open(SVC_PICKLE_FILE, "wb") as file:
        pickle.dump(svc, file)
    with open(X_SCALER_FILE, "wb") as file:
        pickle.dump(X_scaler, file)

    return X_scaler, svc

count = 1
def process_image(
        X_scaler,
        image,
        draw_image,
        svc,
        y_start_stop,
        cell_per_block,
        color_space,
        hist_bins,
        hist_feat,
        hog_channel,
        hog_feat,
        orient,
        pix_per_cell,
        spatial_feat,
        spatial_size,
        scale = 1
        ):

    global count

    t1 = time.time()

    hot_windows, heatmap = find_cars(
        image,
        y_start_stop[0],
        y_start_stop[1],
        scale,
        svc,
        X_scaler,
        color_space=color_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        #hog_channel=hog_channel,
        #spatial_feat=spatial_feat,
        #hist_feat=hist_feat,
        #hog_feat=hog_feat
        )

    #heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    #heat = add_heat(heat, hot_windows)
    labels = label(heatmap)


    #window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    window_img = draw_labeled_bboxes(np.copy(image), labels)

    plt.imshow(window_img)
    show_or_save('heat_{}.png'.format(count))
    count += 1

    t2 = time.time()
    print(round(t2 - t1, 2), 'Seconds to process single image...')
    return window_img

print('Done loading the big Kahunas, process and train!')


#### Next cell #####
## Movie time1

X_scaler, svc = load_classifier()


def process_movie_image(img, debug=False):

    scale = 1.0
    color_space = 'YCrCb'  # Also can be RGB, HSV, LUV, HLS, YUV, YCrCb
    spatial_size = (14, 14)  # results in 3,072 with 16x16, 12,288 with 64x64 and 192 for 8x8;, or x * y * 3
    #  ^^ with minimal training, seems accuracy drops...
    hist_bins = 32  # 16&32 results in 96, 64 in length 192
    orient = 9
    pix_per_cell = 8  # 4 results in 24,300 length feature
    cell_per_block = 2
    hog_channel = ALL_HOG_CHANNELS
    spatial_feat = True
    hist_feat = True
    hog_feat = True  # results in 5,292 with 8x2x9 (cell/block,pix/cell,orient)
    y_start_stop = [400, 670]  # Min and max in y to search in slide_window()

    hot_windows, heat = find_cars(
        img,
        y_start_stop[0],
        y_start_stop[1],
        scale,
        svc,
        X_scaler,
        color_space=color_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        # hog_channel=hog_channel,
        # spatial_feat=spatial_feat,
        # hist_feat=hist_feat,
        # hog_feat=hog_feat
        )

    # Apply threshold to alp remove false positives
    heat = apply_threshold(heat, 0.72)

    # Find final boxes from heatmap using label function
    labels = label(heat)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    if debug:
        return draw_img, heat

    return draw_img

def process_movie():
    fileName = 'project_video.mp4'
    # 'IMG_7462.mp4'
    # 'solidWhiteRight.mp4'
    # 'solidYellowLeft.mp4'

    test_output = 'output/' + fileName
    clip1 = VideoFileClip(fileName).subclip(27,32)
    t1 = time.time()
    output_clip = clip1.fl_image(process_movie_image)
    t2 = time.time()
    print(round(t2 - t1, 2), ' seconds to process movie.')

    output_clip.write_videofile(test_output, audio=False)

#del X_scaler, svc
print('Done loading the movie processing!')


#### Next cell ####
# Testing stuff out

def run_feature_test(test_images, output_file_name='test_feature_{}.png', use_saved_features=False):
    print('Testing out feature extraction by picking randomly from {} images'.format(len(test_images)))

    extract = common_params(extract_features, 'Extracting features')
    features = extract([test_images])

    # if use_saved_features:
    #     with open(FEATURE_PICKLE_FILE, "rb") as file:
    #         features = pickle.load(file)
    # else:
    #     extract = common_params(extract_features, 'Extracting features')
    #     features = extract([test_images])
    #     with open(FEATURE_PICKLE_FILE, "wb") as file:
    #         pickle.dump(features, file)


    if len(features) == 0:
        print('Your function only returns empty feature vectors...')
        return

    print('Feature vector length:', len(features[0]))

    # Create an array stack of feature vectors
    X = np.vstack(features).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    for features_idx, tst_file in enumerate(test_images):
        #features_ind = np.random.randint(0, len(features))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(tst_file)) #test_images[features_ind]))

        name = tst_file.split('/')[-1]
        plt.title('Original Image: {}'.format(name))
        plt.subplot(132)
        plt.plot(X[features_idx])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[features_idx])
        plt.title('Normalized Features: ')
        fig.tight_layout()

        show_or_save(output_file_name.format(name.split('.')[0]))

def run_sliding_windows_test(test_images, output_file_name='sliding_window_test.png'):
    print('Testing out sliding windows')
    jpg_img_idx = np.random.randint(1, 7)
    image = mpimg.imread('./test_images/test{}.jpg'.format(jpg_img_idx))

    windows = slide_window(
        image,
        x_start_stop=[None, None],
        y_start_stop=[None, None],
        xy_window=(128, 128),
        xy_overlap=(0.5, 0.5)
        )

    window_img = draw_boxes(image, windows, color=(155, 55, 255), thick=6)
    plt.imshow(window_img)
    show_or_save(output_file_name)

def run_window_search_test(test_images, output_file_name='search_slide_test_{}.png'):
    print('Running window search test with classification')

    extract = common_params(extract_features, 'Extracting features')

    if SHOULD_TRAIN_CLASSIFIER:
        X_scaler, svc = train_classifier(extract)
    else:
        X_scaler, svc = load_classifier()

    #jpg_img_idx = np.random.randint(1, 7)
    for jpg_img_idx in range(6):
        image = mpimg.imread('./test_images/test{}.jpg'.format(jpg_img_idx + 1)) # mpimg.imread('bbox-example-image.jpg')
        draw_image = np.copy(image)

        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        image = image.astype(np.float32)/255

        process = common_params(process_image, 'Processing image')
        y_start_stop = [400, 670]  # Min and max in y to search in slide_window()
        window_img = process([X_scaler, image, draw_image, svc, y_start_stop])

        plt.imshow(window_img)
        show_or_save(output_file_name.format(jpg_img_idx + 1))

def visualize_feature_extract(output_file_name, viz=False):
    print('Visualizing hog features')
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    extract_img_feature = common_params(single_img_features)

    car_features, car_hog_image = extract_img_feature(image=car_image, viz=viz) ##blech...
    notcar_features, notcar_hog_image = extract_img_feature(image=notcar_image, viz=viz)

    images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
    titles = ['Car Image', 'Car HOG', 'Not Car', 'Not Car HOG']
    fig = plt.figure(figsize=(12,3)) #, dpi=80
    for i, img, in enumerate(images):
        ax = plt.subplot(1, 4, i + 1) # rows, cols, subplotidx
        img_dims = len(img.shape)
        if img_dims < 3:
            ax.imshow(img, cmap='hot')
        else:
            ax.imshow(img)
        ax.set_aspect('auto')
        plt.title(titles[i])

    plt.tight_layout()
    show_or_save(output_file_name)

# Thanks to Q&A
#def visualize_hog(output_file_name='visualize_hog.png'):
#    visualize_feature_extract(output_file_name=output_file_name, viz=True)

def test_process_movie_image(output_file_name='labeled_bbox_{}.png'):
    # Read in a pickle file with bboxes saved
    # Each item in the "all_bboxes" list will contain a
    # list of boxes for one of the images shown above
    #box_list = pickle.load( open( "bbox_pickle.p", "rb" ))

    #X_scaler, svc = load_classifier()

    # Read in image similar to one shown above
    for jpg_img_idx in range(6):
        image = mpimg.imread('./test_images/test{}.jpg'.format(jpg_img_idx + 1))

        draw, heat = process_movie_image(image, debug=True)

        # Find final boxes from heatmap using label function
        labels = label(heat)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heat, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()

        show_or_save(output_file_name.format(jpg_img_idx + 1))

if platform != 'darwin':  # Mac OSX
    print('Only meant for running from command line!  Or maybe not?')
    # TODO: Verify from Jupyter if needed, think it will work...
else:
    test_images = glob.glob('./test_images/*.png')

    # run_feature_test(test_images) #, 'feature_test.png')
    # run_sliding_windows_test(test_images)
    #visualize_hog()
    #visualize_hog(output_file_name='visualize_hog2.png')
    run_window_search_test(test_images)
    del test_images


    test_process_movie_image()
    #process_movie()


print("All done testing!  How's it lookin'!?")


#### Next cell #####