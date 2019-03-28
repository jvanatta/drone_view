#!/usr/bin/env python
# Python 3
import cv2 as cv
import numpy as np

def sun_finder(input_image, overexposure_threshold=251, min_radius=20, max_radius=180, ):
    """ Locate the presence of a likely sun in the image based on overexposed pixels. See readme.md for more details.
    Radius coordinates are in pixels, built for images roughly 1280px wide and a wide angle lens.
    If lots of different sized images are necessary, either resize them upstream or convert the code to relative image sizes.
    Return format: openCV's Keypoint. Not my favorite struct but it's too much for a tuple and too little
    to create a custom class for.
    """
    working_image = np.copy(input_image)
    proportion_required = .5
    erode_amount = 5
    dilate_amount = 10

    working_image = cv.cvtColor(working_image, cv.COLOR_BGR2GRAY)
    _, threshed_image = cv.threshold(working_image, overexposure_threshold, 255, cv.THRESH_BINARY)
    # Try and clean up stray noisy flecks from reflected light, artificial lights, etc.
    threshed_image = cv.erode(threshed_image, cv.getStructuringElement(cv.MORPH_RECT, (erode_amount, erode_amount)))
    threshed_image = cv.dilate(threshed_image, cv.getStructuringElement(cv.MORPH_RECT, (dilate_amount, dilate_amount)))

    # Set up and run OpenCV's SimpleBlobDetector. See
    # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
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

    # This version munging is necessary according to link above
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
    # If there are lots of overexposed image parts, we're likely getting clouds or an entire sky.
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
    return filtered_blobs[-1]


def height_estimation(input_image):
    """ Very quick and dirty height estimation. Use a cubic regression with parameters
    I found from the 31 provided image samples. **Highly likely to be overfit.**
    See readme.md for a picture of the curve fit.
    It's not great to have these embedded in the source like this, but I'm not keen
    on loading them as a config file for something so quick and dirty.
    Units are meters.
    """
    def cubic_function(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d
    a = -1.2429890960756994e-06
    b = 0.002529233793937746
    c = -1.390326640092421
    d = 250.36468897434398

    regression_value = cubic_function(cv.Laplacian(input_image, cv.CV_32F).var(), a, b, c, d)
    # Pin the values between 0 and 2000m, to avoid any polynomial fit weirdness at the extremes
    return min(max(regression_value, 0), 2000)


def night_detector(input_image, black_threshold=10, proportion_required=.6):
    """ What does an image taken from a drone camera taken at night look like? Probably mostly black.
    """
    working_image = np.copy(input_image)
    working_image = cv.cvtColor(working_image, cv.COLOR_BGR2GRAY)
    _, threshed_image = cv.threshold(working_image, black_threshold, 255, cv.THRESH_BINARY_INV)
    black_proportion = float(np.count_nonzero(threshed_image)) / (threshed_image.shape[0] * threshed_image.shape[1])
    return black_proportion > proportion_required


def cloud_detector(input_image, blurry_threshold=18):
    """ Fog looks a lot like blur. The variance of the Laplacian is a nice, easy blur detector.
    See: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    return cv.Laplacian(input_image, cv.CV_32F).var() < blurry_threshold


if __name__ == '__main__':
    print("Module use only")