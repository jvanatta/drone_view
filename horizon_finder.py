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

# Other ideas:
# SVM
# bounding figure over spatial frequency dense areas
# PCA

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

        image_width = input_image.shape[1]
        image_height = input_image.shape[0]

        ## ------------------------------

        # Sobel output is signed, so a step is required before goingto uint8. See
        # https://docs.opencv.org/3.4/d5/d0f/tutorial_py_gradients.html
        input_image = cv.bilateralFilter(input_image, 9, 500, 500)
        blue_channel, green_channel, red_channel = cv.split(input_image)
        target_image = np.uint8(np.absolute(cv.Sobel(green_channel, cv.CV_32F, 0, 1, ksize=3)))

        cutoff_percentile = 99
        cutoff_value = np.percentile(target_image, cutoff_percentile)
        print(cutoff_value, np.max(target_image))
        _, target_image = cv.threshold(target_image, cutoff_value, 255, cv.THRESH_BINARY)

        # Reinterpret as a collection of points:
        #masked_array = np.mask_indices
        selected_points = target_image.nonzero()


        ########
        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])
        ax_h = fig.add_subplot(gs[0])
        #ax_v = fig.add_subplot(gs[1])
        ax_h.scatter(selected_points[1], -1 * selected_points[0], color='b')
        ax_h.set_title("Horizontal")
        ax_h.set_ylim([-1 * image_height, 0])

        #ticks = ticker.FuncFormatter(lambda q, pos: '{0:g}'.format(round(q / (2 * sensor_size_em11_mm[0]))))
        #ax_h.xaxis.set_major_formatter(ticks)
        #ax_h.set_xticks(np.arange(5, pixels_x + 2 * step_size_mtf5[0], 2 * step_size_mtf5[0]))
        #ticks = ticker.FuncFormatter(lambda q, pos: '{0:g}'.format(round(q / (2 * sensor_size_em11_mm[1]))))
        #ax_v.xaxis.set_major_formatter(ticks)
        # ax_v.set_xticks(np.arange(5, pixels_y + 2 * step_size_mtf5[1], 2 * step_size_mtf5[1]))



        # Robustly fit linear model with RANSAC algorithm
        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html


        ransac = linear_model.RANSACRegressor()
        #{'base_estimator': None, 'is_data_valid': None, 'is_model_valid': None, 'loss': 'absolute_loss', 'max_skips': inf, 'max_trials': 100, 'min_samples': None, 'random_state': None, 'residual_threshold': None, 'stop_n_inliers': inf, 'stop_probability': 0.99, 'stop_score': inf}
        ransac.set_params(max_trials=1000, stop_probability=.9999)

        ransac.fit(selected_points[1].reshape(-1, 1), -1 * selected_points[0])
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        #line_y_ransac = ransac.predict(selected_points[1].reshape(-1, 1))
        print("coef??", ransac.estimator_.coef_)
        plt.plot(np.arange(0, image_width), ransac.predict(np.arange(0, image_width).reshape(-1, 1)))

        plt.savefig("output.png")
        cvplot = cv.imread("output.png")

        ## ------------------------------
        while True:
            k = cv.waitKey(1)
            # Press j or f for next image, q or escape to quit
            if k == 'q' or k == 27 or k == 'Q' or k == 1048603 or k == 1048689:
                sys.exit("Quitting")
            elif k == 'j' or k == 102 or k == 'f' or k == 106 or k == 65363:
                break

            cv.imshow("input", input_image)
            cv.imshow("threshed", target_image)
            cv.imshow("plot", cvplot)

    print("All done!")