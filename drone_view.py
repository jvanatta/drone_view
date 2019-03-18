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
    source_file_ext = ".png"

    images_to_process = []
    if os.path.isdir(source_files):
        for filename in glob.glob("{0}/*{1}".format(source_files, source_file_ext)):
            images_to_process.append(filename)
    else:
        images_to_process.append(source_files)

    # Not lexical sort, so pad image filenames with leading zeros!
    images_to_process = sorted(images_to_process)
    print("Starting work on {0} image files...".format(len(images_to_process)))

    for image_filename in images_to_process:
        input_image = cv.imread(image_filename)
        if input_image is None:
            print("Bad input, check your filename: {0}".format(input_image))
            continue
        if (input_image.shape[0] < 50 or input_image.shape[1] < 50):
            print("Skipping image (<50px on edge): {0}".format(input_image))
            continue

        # Some of the samples have a thin black line along the top
        # This dirty data plays hell with edge finding, so just crop it away
        input_image = input_image[2:][::]
        display_image = np.copy(input_image)
        image_width = input_image.shape[1]
        mage_height = input_image.shape[0]

        # False Horizon from horizontal edge detector + RANSAC
        horiz_regression_intercept, horiz_regression_slope = false_horizon.horizontal_edge_regression(input_image)
        if horiz_regression_intercept is not None:
            cv.line(display_image, (0, horiz_regression_intercept),
                    (image_width, int(round(horiz_regression_intercept + horiz_regression_slope * image_width))),
                    thickness=3, color=(0, 50, 255))
        else:
            cv.putText(display_image, "No Horizon Detected", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 50, 255), thickness=4)

        if scene_features.night_detector(input_image):
            cv.putText(display_image, "Night Time", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color=(255, 50, 255), thickness=4)

        if scene_features.cloud_detector(input_image):
            cv.putText(display_image, "Cloudy/Foggy", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, color=(255, 50, 255), thickness=4)

        detected_sun = scene_features.sun_finder(input_image)
        if detected_sun is not None:
            sun_coords = (int(round(detected_sun.pt[0])), int(round(detected_sun.pt[1])))
            sun_radius = int(round(detected_sun.size / 2))
            cv.circle(display_image, sun_coords, sun_radius, color=(130, 0, 180), thickness=5)

        estimated_height = int(round(scene_features.height_estimation(input_image), -1))
        cv.putText(display_image, "Estimated height: {0}m".format(estimated_height),
                   (50, 250), cv.FONT_HERSHEY_SIMPLEX, 1, color=(255, 50, 255), thickness=4)

        while True:
            k = cv.waitKey(1)
            # Press j or f for next image, q or escape to quit
            if k == 'q' or k == 27 or k == 'Q' or k == 1048603 or k == 1048689:
                sys.exit("Quitting")
            elif k == 'j' or k == 102 or k == 'f' or k == 106 or k == 65363:
                break
            cv.imshow("input", display_image)

    print("All done!")
