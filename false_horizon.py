#!/usr/bin/env python
# Python 3
import cv2 as cv
import numpy as np
from sklearn import linear_model

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


if __name__ == '__main__':
    print("Module use only")