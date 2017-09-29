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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# %matplotlib inline #// Jupyter Notebooks only

print('Done importing everything.  System ready to rip!')


#### Next cell ####
# Initial importing of data.

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

del basedir, image_types
print('Printed above size of test sets, also imported cars and notcars')


#### Next Cell ###
# Utility Function and constants

ALL_HOG_CHANNELS = 'ALL'

X_SCALER_FILE = "X_scaler_pickle.p"
SVC_PICKLE_FILE = "svc_pickle.p"

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


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def params_for_feature_extract(
        color_space = 'YCrCb',
        spatial_size = (16, 16),
        hist_bins = 16,
        orient = 9,
        pix_per_cell = 8,
        cell_per_block = 2,
        hog_channel = ALL_HOG_CHANNELS,
        spatial_feat = True,
        hist_feat = True,
        hog_feat = True
        ):
    def call_with_input(input):
        return extract_features(
            input,
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

def load_classifier():
    print('Loading SVC and X_Scaler')
    with open(SVC_PICKLE_FILE, "rb") as file:
        svc = pickle.load(file)
    with open(X_SCALER_FILE, "rb") as file:
        X_scaler = pickle.load(file)

    return X_scaler, svc

print('Loaded all utility functions and some constants')


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
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_image = correct_for_colorspace_or_copy(img, color_space=color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32):  # , bins_range=(0, 256)):
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
        color_space='YCrCb',  # Also can be RGB, HSV, LUV, HLS, YUV, YCrCb
        spatial_size=(24, 24), # next try 32
        hist_bins=24,
        # hist_range=(0, 256),
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
        features.append(img_features)

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
        color_space='YCrCb',
        hist_bins=16,
        hist_feat=True,
        hog_channel=ALL_HOG_CHANNELS,
        hog_feat=True,
        cell_per_block=3,
        orient=9,
        pix_per_cell=7,
        spatial_feat=True,
        spatial_size=(16, 16),
        viz=False
        ):
    img_features = []
    hog_images = []
    feature_image = correct_for_colorspace_or_copy(image, color_space)
    ### NOTE::: Extracting features need to be done in the same order as here
    ### NOTE::: Extracting features need to be done in the same order as here
    ### NOTE::: Extracting features need to be done in the same order as here
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)  # , bins_range=hist_range)
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


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(
        img,
        windows,
        clf,
        scaler,
        color_space='RGB',
        spatial_size=(32, 32),
        hist_bins=32,
        hist_range=(0, 256),
        orient=9,
        pix_per_cell=8,
        cell_per_block=2,
        hog_channel=0,
        spatial_feat=True,
        hist_feat=True,
        hog_feat=True
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

#### Next cell ####
# Testing stuff out

OUTPUT_DIR = './output_images/{}'


def show_or_save(output_file_name='did_not_supply_file_name.png'):
    if platform == 'darwin':
        print('Presuming to be on a mac, saving to file')
        plt.savefig(OUTPUT_DIR.format(output_file_name))
    else:
        print('Presuming to be in Jupyter Notebook, calling show()')
        plt.show()

    plt.figure()

def run_feature_test(test_images, output_file_name='test_feature.png'):
    print('Testing out feature extraction by picking randomly from {} images'.format(len(test_images)))
    features = extract_features(
        test_images,
        color_space='RGB',
        spatial_size=(32, 32),
        hist_bins=32
        # , hist_range=(0, 256)
        )

    if len(features) == 0:
        print('Your function only returns empty feature vectors...')
    else:
        # Create an array stack of feature vectors
        X = np.vstack(features).astype(np.float64)  # , notcar_features
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

        show_or_save(output_file_name)


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

def run_window_search_test(test_images, output_file_name='search_slide_test_{}.png'):
    print('Running window search test with classification')

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = ALL_HOG_CHANNELS  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [400, 670]  # Min and max in y to search in slide_window()

    #X_scaler, svc = load_classifier()
    X_scaler, svc = train_classifier(
        cell_per_block,
        color_space,
        hist_bins,
        hist_feat,
        hog_channel,
        hog_feat,
        orient,
        pix_per_cell,
        spatial_feat,
        spatial_size
        )

    #jpg_img_idx = np.random.randint(1, 7)
    for jpg_img_idx in range(6):
        image = mpimg.imread('./test_images/test{}.jpg'.format(jpg_img_idx + 1)) # mpimg.imread('bbox-example-image.jpg')
        draw_image = np.copy(image)

        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        image = image.astype(np.float32)/255

        t1 = time.time()
        windows = slide_window(
            image,
            x_start_stop=[None, None],
            y_start_stop=y_start_stop,
            xy_window=(96, 96),
            xy_overlap=(0.5, 0.5)
            )

        hot_windows = search_windows(
            image,
            windows,
            svc,
            X_scaler,
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

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        t2 = time.time()

        print(round(t2 - t1, 2), 'Seconds to process single image...')
        plt.imshow(window_img)
        show_or_save(output_file_name.format(jpg_img_idx + 1))

def train_classifier(
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
        should_save=True
        ):

    # Read in cars and notcars
    cars, notcars = from_data_set(num_samples=1000)
    # from_test_images(test_images)

    extract = params_for_feature_extract(
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

    # TODO: When done, should_save should go here, so choice is binary to train or load up.
    # Consider: the very first time, if told to load and nothing exists, don't proceed
    #if should_save:
    car_features = extract(cars)
    notcar_features = extract(notcars)
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
    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    if should_save:
        print('Saving SVC and X_Scaler')
        with open(SVC_PICKLE_FILE, "wb") as file:
            pickle.dump(svc, file)
        with open(X_SCALER_FILE, "wb") as file:
            pickle.dump(X_scaler, file)

    return X_scaler, svc


# Thanks to Q&A
def visualize_hog(output_file_name='visualize_hog.png'):
    print('Visualizing hog features')
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    car_features, car_hog_image = single_img_features(image=car_image, viz=True)
    notcar_features, notcar_hog_image = single_img_features(image=notcar_image, viz=True)

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


if platform != 'darwin':  # Mac OSX
    print('Only meant for running from command line!  Or maybe not?')
    # TODO: Verify from Jupyter if needed, think it will work...
else:
    test_images = glob.glob('./test_images/*.png')

    run_feature_test(test_images, 'feature_test.png')
    run_sliding_windows_test(test_images)
    visualize_hog()
    visualize_hog(output_file_name='visualize_hog2.png')
    run_window_search_test(test_images)

    del test_images

print("All done testing!  How's it lookin'!?")

#### Next cell ###
