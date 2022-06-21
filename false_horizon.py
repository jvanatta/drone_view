#!/usr/bin/env python
# Python 3
from typing import Tuple

import cv2 as cv
import numpy as np
from sklearn import linear_model

def horizontal_edge_regression(input_image: np.ndarray, cutoff_percentile: float = 97) -> Tuple[float, float]:
    """Find the slope and intercept of a best guess at the false horizon.
    For details and graphical examples, see readme.md
    cutoff_percentile determines how many of the detected horizontal edges
    to feed into the RANSAC fitting. Making it 95 or 99 will definitely
    change the outcome of marginal cases.
    """
    working_image = np.copy(input_image)

    # Sobel output is signed, so a step is required before goingto uint8. See
    # https://docs.opencv.org/3.4/d5/d0f/tutorial_py_gradients.html
    # Using the grayscale image works too, but the green channel seems most reliable
    working_image = cv.GaussianBlur(working_image, (11, 11), 100)
    blue_channel, green_channel, red_channel = cv.split(working_image)
    working_image = np.uint8(np.absolute(cv.Sobel(green_channel, cv.CV_32F, 0, 1, ksize=3)))

    cutoff_value = np.percentile(working_image, cutoff_percentile)
    _, threshed_image = cv.threshold(working_image, cutoff_value, 255, cv.THRESH_BINARY)

    # Reinterpret the binary image as a collection of points to use scikit's RANSAC
    # So a TRUE pixel at x=25, y=21 becomes a sample, (25, 21)
    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html
    selected_points = threshed_image.nonzero()
    if len(selected_points[0] > 0):
        ransac = linear_model.RANSACRegressor()
        # Squared loss is a big improvement over absolute loss in the samples tested
        ransac.set_params(max_trials=1000, stop_probability=.9999, loss='squared_error')
        ransac.fit(selected_points[1].reshape(-1, 1), -1 * selected_points[0])

        # Multiply by -1 to convert from "plot" coordinates to "image" coordinates
        ransac_intercept = -1 * int(round(ransac.estimator_.intercept_))
        ransac_slope = -1 * ransac.estimator_.coef_[0]
        return [ransac_intercept, ransac_slope]
    else:
        return [None, None]


if __name__ == '__main__':
    print("Module use only")