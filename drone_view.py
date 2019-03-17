#!/usr/bin/env python
# Python 3
import glob
import os
import sys

import cv2 as cv
import numpy as np

import false_horizon
import scene_features

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

        horiz_regression_intercept, horiz_regression_slope = false_horizon.horizontal_edge_regression(np.copy(input_image))
        if horiz_regression_intercept is not None:
            cv.line(display_image, (0, horiz_regression_intercept),
                    (image_width, int(round(horiz_regression_intercept + horiz_regression_slope * image_width))),
                    thickness=3, color=(0, 0, 255))

        if scene_features.night_detector(np.copy(input_image)):
            cv.putText(display_image, "Night Time", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color=(80, 200, 255), thickness=4)

        if scene_features.cloud_detector(np.copy(input_image)):
            cv.putText(display_image, "Cloudy/Foggy", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, color=(80, 200, 255), thickness=4)

        detected_sun = scene_features.sun_finder(np.copy(input_image))
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