##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

1) Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
2) Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
3)  Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
4)  Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/labeled_bbox_6.png
[image2]: ./output_images/visualize_hog2.png
[image3]: ./output_images/sliding_window_test.png
[image4]: ./output_images/search_slide_test.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/features.png
[image7]: ./examples/output_bboxes.png
[video1]: ./output/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####0. Initial setup
I recently discovered the super-happy-puppy-baby-monkey awesome PyCharm is indeed a rather pleasant experience for coding as it is for Java and Javascript with IntelliJ.  My initial idea was to write everything in a Python script, using refactorings that I'm used to in Java to split up the functions across files/etc sanely, and then transfer it over to a Jupyter Notebook. 
 
 For one, I have ran out of time to transfer to Jupyter, but at least some of that basic structure is still there (with the ##### Next Cell  ## designations).  Taking lessons learned from prior Jupyter notebooks, I took pains to ensure local variables don't leak, deleting them manually as necessary.   

  Unfortunately, I took this cause of avoiding global vars too far and created functions-of-functions to encapsulate my variables.  Really enjoyed to have discovered Python does structuring similar to Javacript, however, not as clean of a syntax.  I should have just had all the parameters as a constants object that can be passw round
####1. Explain how (and identify where in your code) you extracted HOG features from the training images.
Since my code is really in flux still, I'll try to avid using line-numbers.  I renamed the single_img_features method frm the lecture videos as that was the initial place I had HOG feature extraction (and spatial binning) working.  However, upon following the lecture notes, I proceeded to use the extract HOG-once trick, even though I noticed a small degredation on the smaller subset of images I tested against.
  
  I renamed the original hog feature to be `mcv_init_single_img_features`, searching for that will bring you to what you seek. 
  A method called process_movie_image contains the approach used in the video, 

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.
I initially tried using the visualize_hog feature I had written (whose remants can still be found).  An unfortunate side effect of that was that I unfortunately hacked the `mcv_init_single_img_features` method to take a parameter for visualization (as found in the lecture notes) that would return the raw HOG image for visualization.  The problem with it is it omplicated the functor I had created to try to centralize the parameters into one location. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM as found in the course.  It was a conflicting choice for me as to which classifier to use as I have a personal connection to both decision trees and SVMs.  My grad school advisor's first student was Corinna Cortes, who co-rotoe one of the first papaers of SVMs with Vladimir Vapnick and is now head of Google Research NYC.  The connection I had to decision trees is that while I worked at Amazon, I met someone wrote a random-forest-as-a-service (that should now be included in the machine-learning AWS services) who informed me how great they were, despite their simpleness compared to SVMs.  


In the end, I was surprised to discover after several iterations that Random Forests didn't classify well at all. I also didn't find much discussion in the forum about alternatives to SVMs, so I stuck with them.

I discovered a very high-C to fit to the training data work since a lot of the data was hard negative mining from the video itself.  When I added the CrowdAi set, I reduced it to avoid overfitting to the more generalized data set.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My preferred idea to improve performance is to add a RandomForestClassifier, and something like XGBoost or whatever it is is the cool cats use these days, and then use an ensemble of them.  When I briefly tried to use purely RandomForestClassifier, absolutely nothing seemed to have come out of it.  
I also completely ran out of memory when trying to increase the training set size to the full 70,000 provided by the Udacity CrowdAI project.  Can spend more time improving on that. 


