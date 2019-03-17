#!/usr/bin/env python
# Python 3
import cv2 as cv
import numpy as np


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
    print("Module use only")