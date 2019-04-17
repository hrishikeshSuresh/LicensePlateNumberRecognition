# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:00:10 2019

@author: Hrishikesh S.

DEVELOPER COMMENTS : # for explanation
                     ## for removing code
"""

# cd "Desktop/Third Year/Machine Learning/Project"

import numpy as np
import cv2
import os
# For FFT2
import scipy.fftpack  
import pandas as pd
##from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
##import cca2
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure
from skimage.measure import regionprops
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils

##import localization

files = os.listdir("data")

csv_files = pd.read_csv("data/trainVal.csv")


# cv2.IMREAD_COLOR : loads color image, use 1
# cv2.IMREAD_GRAYSCALE : loads image in grayscale mode, use 0
# cv2.IMREAD_UNCHANGED : loads image as such including alpha channel, use -1
def image_extraction(csv_files, channel):
    i = 0
    raw_data = []
    labels = []
    for index, row in csv_files.iterrows():
        i = i + 1
        file = row['image_path']
        label = row['lp']
        op_filename = "images/grayscale/" + file.split(sep = '/')[1] + "/" + file.split(sep = '/')[2].replace(".png", ".jpg")
        print(op_filename)
        ip_filename = "data/"+ file.split(sep = '/')[1] + '/' + file.split(sep = '/')[2]
        print(ip_filename)
        img = cv2.imread(ip_filename, channel)
        raw_data.append(img)
        labels.append(label)
        ##plt.imsave(op_filename, img)
    print(i)
    return raw_data, labels

# imclearborder definition
def imclearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    param_ext, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

# bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    param_ext, contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

def homomorphic_filter(csv_files):
    filtered_data = []
    labels = []
    i = 0
    for index, row in csv_files.iterrows():
        i = i + 1
        try:
            file = row['image_path']
            label = row['lp']
            filename = "data/"+ file.split(sep = '/')[1] + '/' + file.split(sep = '/')[2]
            print(i, filename)
            img = cv2.imread(filename, 0)
            # Number of rows and columns
            rows = img.shape[0]
            cols = img.shape[1]
            # Remove some columns from the beginning and end
            img = img[:, 59:cols-20]
            # Number of rows and columns
            rows = img.shape[0]
            cols = img.shape[1]
            # Convert image to 0 to 1, then do log(1 + I)
            imgLog = np.log1p(np.array(img, dtype="float") / 255)
            # Create Gaussian mask of sigma = 10
            M = 2*rows + 1
            N = 2*cols + 1
            sigma = 10
            (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
            centerX = np.ceil(N/2)
            centerY = np.ceil(M/2)
            gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2
            # Low pass and high pass filters
            Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
            Hhigh = 1 - Hlow
            # Move origin of filters so that it's at the top left corner to
            # match with the input image
            HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
            HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())
            # Filter the image and crop
            If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
            Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
            Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))
            # Set scaling factors and add
            gamma1 = 0.3
            gamma2 = 1.5
            Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]
            # Anti-log then rescale to [0,1]
            Ihmf = np.expm1(Iout)
            Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
            Ihmf2 = np.array(255*Ihmf, dtype="uint8")
            # Threshold the image - Anything below intensity 65 gets set to white
            Ithresh = Ihmf2 < 65
            Ithresh = 255*Ithresh.astype("uint8")
            # Clear off the border.  Choose a border radius of 5 pixels
            Iclear = imclearborder(Ithresh, 5)
            # Eliminate regions that have areas below 120 pixels
            Iopen = bwareaopen(Iclear, 120)
            # Show all images
            ##cv2.imshow('Original Image', img)
            ##cv2.imshow('Homomorphic Filtered Result', Ihmf2)
            ##cv2.imshow('Thresholded Result', Ithresh)
            ##cv2.imshow('Opened Result', Iopen)
            ##cv2.waitKey(0)
            ##cv2.destroyAllWindows()
            filtered_data.append(Iopen)
            labels.append(label)
        except:
            pass
    return filtered_data, labels

# gray scale data
##X_gray_scale, y_gray_scale = image_extraction(csv_files, 0)
# data with rgb
##X_rgb, y_rgb = image_extraction(csv_files, 1)

# an example
license_plate = imread("data/crop_h1/I00000.png", as_grey = True)/255.0 
print(license_plate.shape)

# see the difference between gray scale and binary image
gray_car_image = license_plate * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap = "gray")
# threshold_otsu is an algorithm to reduce grayscale image to binary image
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image, cmap = "gray")

print(binary_car_image)

filtered_data, labels = homomorphic_filter(csv_files)

cv2.imshow('Homomorphic filtered output', filtered_data[100])
cv2.waitKey(0)
cv2.destroyAllWindows()

# character segmentation
##labelled_plate = measure.label(license_plate)
##fig, ax1 = plt.subplots(1)
##ax1.imshow(license_plate, cmap = "gray")
##character_dimensions = (0.35*license_plate.shape[0],
##                        0.60*license_plate.shape[0],
##                        0.05*license_plate.shape[1], 
##                        0.15*license_plate.shape[1])
##min_height, max_height, min_width, max_width = character_dimensions

##cord = []
##counter = 0
##column_list = []
##for regions in regionprops(labelled_plate):
    ##y0, x0, y1, x1 = regions.bbox
    ##region_height = y1 - y0
    ##region_width = x1 - x0
    ##if(region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width):
        ##roi = license_plate[y0:y1, x0:x1]
        ##rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", linewidth=2, fill=False)
        ##ax1.add_patch(rect_border)
        ##resized_char = resize(roi, (20, 20))
        
# extract the Value component from the HSV color space and apply adaptive thresholding to reveal the characters on the license plate
##V = cv2.split(cv2.cvtColor(license_plate, cv2.COLOR_BGR2HSV))[2]
##T = threshold_local(V, 29, offset=15, method="gaussian")
##thresh = (V > T).astype("uint8") * 255
##thresh = cv2.bitwise_not(thresh)
 
# resize the license plate region to a canonical size
##plate = imutils.resize(plate, width=400)
##thresh = imutils.resize(thresh, width=400)
##cv2.imshow("Thresh", thresh)


# perform a connected components analysis and initialize the mask to store the locations of the character candidates
##labels = measure.label(thresh, neighbors=8, background=0)
##charCandidates = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
##for label in np.unique(labels):
    # if this is the background label, ignore it
	##if label == 0:
		##continue
    # otherwise, construct the label mask to display only connected components for the current label, then find contours in the label mask
	##labelMask = np.zeros(thresh.shape, dtype="uint8")
	##labelMask[labels == label] = 255
	##cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	##cnts = cnts[0] if imutils.is_cv2() else cnts[1]