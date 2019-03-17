#!/usr/bin/env python
# Python 3
import glob
import os
import sys
from sklearn import linear_model, datasets

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

import cv2 as cv
import numpy as np

def horizontal_edge_regression(input_image):
    cutoff_percentile = 97
    image_width = input_image.shape[1]
    image_height = input_image.shape[0]

    horizontal_edge_display = np.copy(input_image)
    # Sobel output is signed, so a step is required before goingto uint8. See
    # https://docs.opencv.org/3.4/d5/d0f/tutorial_py_gradients.html
    blurred_image = cv.GaussianBlur(horizontal_edge_display, (11, 11), 100)
    #grayscale_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)
    blue_channel, green_channel, red_channel = cv.split(blurred_image)
    edges_image = np.uint8(np.absolute(cv.Sobel(green_channel, cv.CV_32F, 0, 1, ksize=3)))

    cutoff_value = np.percentile(edges_image, cutoff_percentile)
    #print("Percentile cutoff, ", cutoff_value, " max ", np.max(edges_image))
    _, edges_image = cv.threshold(edges_image, cutoff_value, 255, cv.THRESH_BINARY)

    # Reinterpret as a collection of points to use scikit's RANSAC:
    selected_points = edges_image.nonzero()
    # Robustly fit linear model with RANSAC algorithm
    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html
    if len(selected_points[0] > 0):
        ransac = linear_model.RANSACRegressor()
        ransac.set_params(max_trials=1000, stop_probability=.9999, loss='squared_loss')
        ransac.fit(selected_points[1].reshape(-1, 1), -1 * selected_points[0])

        # Multiply by -1 to convert from "plot" coordinates to "image" coordinates
        ransac_intercept = -1 * int(round(ransac.estimator_.intercept_))
        ransac_slope = -1 * ransac.estimator_.coef_[0]

        return [ransac_intercept, ransac_slope]
    else:
        return [None, None]


''' Return format: openCV's Keypoint. Not my favorite struct but it's too much for a tuple and too little
    to create a custom class for'''
def sun_finder(input_image):
    brightness_threshold = 251
    min_radius = 20
    max_radius = 180
    proportion_required = .5
    erode_amount = 5
    dilate_amount = 10

    grayscale_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    _, threshed_image = cv.threshold(grayscale_image, brightness_threshold, 255, cv.THRESH_BINARY)
    threshed_image = cv.erode(threshed_image, cv.getStructuringElement(cv.MORPH_RECT, (erode_amount, erode_amount)))
    threshed_image = cv.dilate(threshed_image, cv.getStructuringElement(cv.MORPH_RECT, (dilate_amount, dilate_amount)))

    blob_detector_param = cv.SimpleBlobDetector_Params()
    blob_detector_param.filterByColor = False
    blob_detector_param.filterByArea = True
    blob_detector_param.minArea = int(round(min_radius**2 * np.pi))
    blob_detector_param.maxArea = int(round(max_radius**2 * np.pi))
    blob_detector_param.filterByCircularity = True
    blob_detector_param.minCircularity = .2
    blob_detector_param.filterByInertia = True
    blob_detector_param.minInertiaRatio = .3
    blob_detector_param.filterByConvexity = False

    # This version munging is necessary according to link below
    # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    blob_detector = None
    ver = (cv.__version__).split('.')
    if int(ver[0]) < 3:
        blob_detector = cv.SimpleBlobDetector(blob_detector_param)
    else:
        blob_detector = cv.SimpleBlobDetector_create(blob_detector_param)

    blobs = blob_detector.detect(threshed_image)
    if blobs is None:
        return None

    # Check how much this overexposed section is relative to the entire image.
    # We only want one sun location no matter what, so sort and return the largest.
    filtered_blobs = []
    total_masked = np.count_nonzero(threshed_image)
    for blob in blobs:
        blob_area = np.pi * (blob.size / 2)**2
        if (blob_area / total_masked) > proportion_required:
            filtered_blobs.append(blob)

    if len(filtered_blobs) == 0:
        return None

    filtered_blobs = sorted(filtered_blobs, key=lambda x: x.size)
    return filtered_blobs[0]

def night_detector(input_image, black_threshold=10, proportion_required=.6):
    grayscale_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    _, threshed_image = cv.threshold(grayscale_image, black_threshold, 255, cv.THRESH_BINARY_INV)
    black_proportion = float(np.count_nonzero(threshed_image)) / (threshed_image.shape[0] * threshed_image.shape[1])
    return black_proportion > proportion_required

''' Fog looks a lot like blur. The variance of the Laplacian is a good easy blur detector.
    See: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/'''
def cloud_detector(input_image, blurry_threshold=18):
    return cv.Laplacian(input_image, cv.CV_32F).var() < blurry_threshold


if __name__ == '__main__':
    source_files = "images"

    images_to_process = []
    if os.path.isdir(source_files):
        for filename in glob.glob("{0}/*.png".format(source_files)):
            images_to_process.append(filename)
    else:
        images_to_process.append(source_files)

    images_to_process = sorted(images_to_process)
    print("Starting work on {0} image files...".format(len(images_to_process)))

    for image_filename in images_to_process:
        input_image = cv.imread(image_filename)
        if input_image is None:
            sys.exit("Bad input, check your filename: {0}".format(input_image))

        # Some of the samples have a thin black line along the top
        # This dirty data plays hell with edge finding, so just crop it away
        input_image = input_image[2:][::]
        display_image = np.copy(input_image)
        image_width = input_image.shape[1]
        mage_height = input_image.shape[0]

        horiz_regression_intercept, horiz_regression_slope = horizontal_edge_regression(np.copy(input_image))
        if horiz_regression_intercept is not None:
            cv.line(display_image, (0, horiz_regression_intercept),
                    (image_width, int(round(horiz_regression_intercept + horiz_regression_slope * image_width))),
                    thickness=3, color=(0, 0, 255))

        if night_detector(np.copy(input_image)):
            cv.putText(display_image, "Night Time", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color=(80, 200, 255), thickness=4)

        if cloud_detector(np.copy(input_image)):
            cv.putText(display_image, "Cloudy/Foggy", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, color=(80, 200, 255), thickness=4)

        detected_sun = sun_finder(np.copy(input_image))
        if detected_sun is not None:
            sun_coords = (int(round(detected_sun.pt[0])), int(round(detected_sun.pt[1])))
            sun_radius = int(round(detected_sun.size / 2))
            cv.circle(display_image, sun_coords, sun_radius, color=(200, 0, 255), thickness=5)

        while True:
            k = cv.waitKey(1)
            # Press j or f for next image, q or escape to quit
            if k == 'q' or k == 27 or k == 'Q' or k == 1048603 or k == 1048689:
                sys.exit("Quitting")
            elif k == 'j' or k == 102 or k == 'f' or k == 106 or k == 65363:
                break

            cv.imshow("input", display_image)
            #cv.imshow("threshed", threshed_image)

    print("All done!")