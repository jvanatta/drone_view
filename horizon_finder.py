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
    print("Percentile cutoff, ", cutoff_value, " max ", np.max(edges_image))
    _, edges_image = cv.threshold(edges_image, cutoff_value, 255, cv.THRESH_BINARY)

    # Reinterpret as a collection of points:
    selected_points = edges_image.nonzero()
    if len(selected_points[0] > 0):
        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax_h = fig.add_subplot(gs[0])
        ax_h.scatter(selected_points[1], -1 * selected_points[0], color='b')
        ax_h.set_title("Filtered Points")
        ax_h.set_ylim([-1 * image_height, 0])

        # Robustly fit linear model with RANSAC algorithm
        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html
        ransac = linear_model.RANSACRegressor()
        ransac.set_params(max_trials=1000, stop_probability=.9999, loss='squared_loss')
        ransac.fit(selected_points[1].reshape(-1, 1), -1 * selected_points[0])

        # Multiply by -1 to convert from "plot" coordinates to "image" coordinates
        ransac_intercept = -1 * int(round(ransac.estimator_.intercept_))
        ransac_slope = -1 * ransac.estimator_.coef_[0]

        return (ransac_intercept, ransac_slope)
    else:
        return (None, None)


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

        horiz_regression_intercept, horiz_regression_slope = horizontal_edge_regression(input_image)
        if horiz_regression_intercept is not None:
            cv.line(display_image, (0, horiz_regression_intercept),
                    (image_width, int(round(horiz_regression_intercept + horiz_regression_slope * image_width))),
                    thickness=3, color=(0, 0, 255))

        while True:
            k = cv.waitKey(1)
            # Press j or f for next image, q or escape to quit
            if k == 'q' or k == 27 or k == 'Q' or k == 1048603 or k == 1048689:
                sys.exit("Quitting")
            elif k == 'j' or k == 102 or k == 'f' or k == 106 or k == 65363:
                break

            cv.imshow("input", display_image)

    print("All done!")