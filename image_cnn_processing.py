# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:00:10 2019

@author: Hrishikesh S.

DEVELOPER COMMENTS : # for explanation
                     ## for removing code
                     If Github repository downloaded, go to line 470
"""

# cd "Desktop/Third Year/Machine Learning/Project"

import numpy as np
import cv2
import os
# For FFT2
import scipy.fftpack  
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from skimage.io import imread
from skimage.filters import threshold_otsu
import pickle
from PIL import Image
import random
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

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

# character segmentation algorithms
# MSER Method (deprecated)
# only draws contours around the alphabets
# Maximally Stable External Region extractor
def MSER():
    img = cv2.imread('data/crop_h1/I00000.png')
    mser = cv2.MSER_create()
    # Resize the image so that MSER can work better
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    regions = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    cv2.polylines(vis, hulls, 1, (0,255,0)) 
    cv2.namedWindow('img', 0)
    cv2.imshow('img', vis)
    while(cv2.waitKey()!=ord('q')):
        continue
    cv2.destroyAllWindows()
    cv2.imshow('Homomorphic filtered output', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cca v1
    image = cv2.imread('data/crop_h1/I00000.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # threshold the image to reveal light regions in the
    # blurred image
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8") 
    # loop over the unique components
    for label in np.unique(labels):
    	# if this is the background label, ignore it
    	if label == 0:
    		continue 
    	# otherwise, construct the label mask and count the
    	# number of pixels 
    	labelMask = np.zeros(thresh.shape, dtype="uint8")
    	labelMask[labels == label] = 255
    	numPixels = cv2.countNonZero(labelMask) 
    	# if the number of pixels in the component is sufficiently
    	# large, then add it to our mask of "large blobs"
    	if numPixels > 300:
    		mask = cv2.add(mask, labelMask)
    cv2.imshow('Filtered output', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def get_component(data,i,j):
	#returns a single component which is in the same component as i,j in the pixel
	#set data[i][j] = 0 so that it will not go to an infinite loop
	data[i][j] = 0
	req = [(i,j)]
	itr = 0
	while(itr < len(req)):
		x = req[itr][0]
		y = req[itr][1]
		itr+=1
		if(x > 0):
			if(data[x-1][y] == 255):
				data[x-1][y] = 0
				req.append((x-1,y))
			if(y > 0):
				if(data[x-1][y-1] == 255):
					data[x-1][y-1] = 0
					req.append((x-1,y-1))
			if(y < len(data[0]) - 1):
				if(data[x-1][y+1] == 255):
					data[x-1][y+1] = 0
					req.append((x-1,y+1))
		if(y > 0):
			if(data[x][y-1] == 255):
				data[x][y-1] = 0
				req.append((x,y-1))
		if(x < len(data)-1):
			if(data[x+1][y] == 255):
				data[x+1][y] = 0
				req.append((x+1,y))
			if(y > 0):
				if(data[x+1][y-1] == 255):
					data[x+1][y-1] = 0
					req.append((x+1,y-1))
			if(y< len(data[0]) - 1):
				if(data[x+1][y+1] == 255):
					data[x+1][y+1] = 0
					req.append((x+1,y+1))
		if(y < len(data[0]) - 1):
			if(data[x][y+1] == 255):
				data[x][y+1] = 0
				req.append((x,y+1))


	return req

def get_segments(data):
	#sends an array of segmented images, provided the data has only 0->black and 255->white.
	#the required output
	segments = list()
	for i in range(len(data)):
		#for every row in the image 
		for j in range(len(data[i])):
			#for every cell in a row
			if(data[i][j] == 255):
				segments.append(get_component(data,i,j))
	return segments 

def print_segments(segments):
	individual = []
	for segment in segments:
		#initialize to a very large value
		top_left_row = 100000000
		top_left_col = 100000000
		bottom_right_row = -1
		bottom_right_col = -1
		#get the top left and bottom right co-ordinates to decide the size of the component
		for (x,y) in segment:
			top_left_col = min(top_left_col, y)
			top_left_row = min(top_left_row, x)
			bottom_right_col = max(bottom_right_col, y)
			bottom_right_row = max(bottom_right_row, x)
		#create a new image with the determined size
		#+20 only to be on the safer side
		#if you are modifying, it has to be atleast +1
		img = Image.new('L',(bottom_right_row - top_left_row + 1, bottom_right_col - top_left_col + 1))
		pixel = img.load()
		#initialize all the pixels to be black
		for i in range(bottom_right_row - top_left_row + 1):
			for j in range(bottom_right_col - top_left_col + 1):
				pixel[i, j] = 0
		#for all the co-ordinates in the component, set it to white
		for i in segment:
			##print(i[0] - top_left_row," and ",i[1] - top_left_col)
			pixel[i[0] - top_left_row, i[1] - top_left_col] = 255
		#print the segment
		##img.show()
		individual.append(img)
	return individual

def convert_image_to_numpy(individual):
    characters = []
    for i in individual:
        inter_mediate = np.array(i)
        characters.append(inter_mediate)    
    for i in characters:
        cv2.imshow('CHAR', i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return characters

# gray scale data
##X_gray_scale, y_gray_scale = image_extraction(csv_files, 0)
# data with rgb
##X_rgb, y_rgb = image_extraction(csv_files, 1)

# an example to show difference between grayscale image and binary image
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

# to show an image
cv2.imshow('Homomorphic filtered output', filtered_data[578])
cv2.waitKey(0)
cv2.destroyAllWindows()

# making a copy
copy_filtered_data = filtered_data

# saving / printing filtered data
def save_filtered_data(copy_filtered_data, labels):
    for i in range(0, len(copy_filtered_data)):
        ##cv2.imwrite("images/filtered/" + labels[i] + "-" + str(i) + ".png", copy_filtered_data[i])
        cv2.imshow(str(labels[i]) + " " + str(i), copy_filtered_data[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return

def filtered_image_extraction(files):
    clean_data = []
    labels = []
    for file in files:
        img = cv2.imread("images/filtered/" + file, 1)
        label = file.split(sep = "-")[0]
        clean_data.append(img)
        labels.append(label)
    return clean_data, labels

filtered_files = os.listdir("images/filtered")
clean_data, clean_labels = filtered_image_extraction(filtered_files)
#m will be sent as reference and BE AWARE, once you call this m will be BLACK every where.
#so if you want to store the original image some where make sure to copy in another variable
segments_list = []
for m in filtered_data:
    corner_y = np.shape(m)[1] - 1
    corner_x = np.shape(m)[0] - 1
    get_component(m, 0, 0)
    get_component(m, 0, corner_y)
    get_component(m, corner_x, 0)
    get_component(m, corner_x, corner_y)
    segments_list.append(get_segments(m))
#print("segments is ",segments)
individual_list = []
for segments in segments_list:
    individual_list.append(print_segments(segments))
#individual can be used for further processing
# converting PIL image to numpy arrays
X = []
for plate in individual_list:
    for char in plate:
        X.append(np.array(char))

# for labels
a = labels[0]
for i in a:
    print(i)

Y = []
for i in labels:
    for j in i:
        Y.append(j)

copy_X = X

# collecting index row-wise removal
index = []
for i in range(0, len(copy_X)):
    if(np.shape(copy_X[i])[0] > 40 or np.shape(copy_X[i])[0] < 15):
        index.append(i)

# collecting index column-wise removal
index = []
for i in range(0, len(copy_X)):
    if(np.shape(np.shape(copy_X[i])[1] < 15 or np.shape(copy_X[i])[1] > 100)):
        index.append(i)

# display
for i in index:
    cv2.imshow(str(i), copy_X[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def noise_removal(copy_X):
    factor = 0
    for i in index:
        del copy_X[i - factor]
        factor = factor + 1
    ##return copy_X
    for i in range(0, len(copy_X)):
        file = "images/individual/" +  str(i)  + ".png"
        cv2.imwrite(file, copy_X[i])
        ##cv2.imshow(str(i), copy_X[i])
        ##cv2.waitKey(0)
        ##cv2.destroyAllWindows()

def flip_and_rotate(copy_X):
    clean = []
    individual_files = os.listdir('images/individual')
    for i in range(0, len(individual_files)):
        img = cv2.imread('images/individual/' + individual_files[i])
        ##img = copy_X[i]
        img = cv2.flip(img, 1)
        clean.append(img)
    ##cv2.imshow(str(i), clean[0])
    ##cv2.waitKey(0)
    ##cv2.destroyAllWindows()
    for i in range(0, len(clean)):
        cv2.imwrite('images/clean/' + str(i) + '.png', clean[i])

# if repository download, execute from here
# load all binary image from segregated
# grayscale load done to accomodate laoding of image as a 2d array
def final_extraction(folder_list):
    X = []
    Y = []
    # iterate through each folder
    for folder in folder_list:
        file_list = os.listdir('images/segregated/' + folder)
        # iterate over all files in a folder
        for file in file_list:
            img = cv2.imread('images/segregated/' + folder + '/' + file, 0)
            X.append(img)
            Y.append(folder)
    return X, Y

# listing all folders
folder_list = os.listdir('images/segregated/')
# loading the images
X, Y = final_extraction(folder_list)

# determine maximum row & column size
# to know the size to which we have to pad
def determine_max_row_and_column_size(data):
    max_row_size = 0
    max_col_size = 0
    for i in data:
        size = np.shape(i)
        if(size[0] > max_row_size):
            max_row_size = size[0]
        if(size[1] > max_col_size):
            max_col_size = size[1]
    return max_row_size, max_col_size

pad_x, pad_y = determine_max_row_and_column_size(X)

# padding by resizing
# we can also do a zero padding
def image_padding_by_resize(data, pad_x, pad_y):
    out = []
    for i in data:
        u = cv2.resize(i, (pad_x, pad_y))
        out.append(u)
    return out

# padding by resize
X_final = image_padding_by_resize(X, pad_x, pad_y)

# encode class values as integers
# one hot encoding
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers into categorical values 
# by converting it into a bit array form (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

reshape_X = np.reshape(X_final, (-1, 40, 140, 1))

def build_model(data):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(40, 140, 1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(29, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    return model

def train_model(model, train_X, train_y, batch_size, epochs):
    model = model.fit(train_X, train_y, batch_size=batch_size,epochs=epochs)
    return model

LPR_model = build_model(X)

LPR_model.summary()

LPR_model = train_model(LPR_model, reshape_X, dummy_y, 100, 10)