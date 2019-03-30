# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:00:10 2019

@author: Hrishikesh S
"""

# cd "Desktop/Third Year/Machine Learning/Project"

import numpy as np
import cv2
import os
import pandas as pd
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

files = os.listdir("data")

csv_files = pd.read_csv("data/trainVal.csv")

##image = load_img("./data/crop_m1/100001.png")
##plt.imshow(image)

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

# gray scale data
X_mono, y_mono = image_extraction(csv_files, 0)
# data with rgb
X_rgb, y_rgb = image_extraction(csv_files, 1)