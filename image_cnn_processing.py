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

files = os.listdir("data")

csv_files = pd.read_csv("data/trainVal.csv")

# cv2.IMREAD_COLOR : loads color image, use 1
# cv2.IMREAD_GRAYSCALE : loads image in grayscale mode, use 0
# cv2.IMREAD_UNCHANGED : loads image as such including alpha channel, use -1
def image_extraction(csv_files):
    i = 0
    raw_data = []
    for index, row in csv_files.iterrows():
        i = i + 1
        file = row['track_id']
        print("images/grayscale/" + file.split(sep = '/')[1] + "/" + file.split(sep = '/')[2].replace(".png", ".jpg"))
        img = cv2.imread(file, 0)
        raw_data.append(img)
        cv2.imshow('image', img)
        return img
        cv2.imwrite("images/grayscale/" + file.split(sep = '/')[1] + "/" + file.split(sep = '/')[2].replace(".png", ".jpg"), img)
    print(i)
    return img

raw_data = image_extraction(csv_files)