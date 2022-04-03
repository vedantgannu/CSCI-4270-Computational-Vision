#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
import sys
import math


if len(sys.argv) != 5:
    print("Incorrect number of arguments")
    sys.exit(1)
INPUT_IMAGE = sys.argv[1]
OUTPUT_IMAGE = sys.argv[2]
M_DIM = int(sys.argv[3])
N_DIM = int(sys.argv[4])


img = cv2.imread(INPUT_IMAGE)
old_shape = img.shape

min_dimension_loc = np.argmin(img.shape[0:2])
if img.shape[0] < img.shape[1]:#If the number of rows is the smallest dimension
    #Adjust_val used to trim equal number of pixels off from which dimension is largest
    #If can't distribute equally, then remove x // 2 pixels from left side and x // 2 + 1 from right side
    adjust_val = np.abs(img.shape[0] - img.shape[1]) / 2
    img = img[:,int(adjust_val):img.shape[1]-math.ceil(adjust_val)]
    print("Image {} cropped at (0, {}) and ({}, {})".format(INPUT_IMAGE, int(adjust_val), img.shape[0]-1, old_shape[1]-math.ceil(adjust_val)-1))
elif img.shape[0] > img.shape[1]:#If number of columns is smallest dimension
    #Adjust_val used to trim equal number of pixels off from which dimension is largest
    #If can't distribute equally, then remove x // 2 pixels from top side and x // 2 + 1 from bottom side
    adjust_val = np.abs(img.shape[0] - img.shape[1]) / 2
    img = img[int(adjust_val):img.shape[0]-math.ceil(adjust_val),:,:]
    print("Image {} cropped at ({}, 0) and ({}, {})".format(INPUT_IMAGE, int(adjust_val), old_shape[0]-math.ceil(adjust_val)-1),img.shape[1]-1)


old_shape = img.shape
imgResized = cv2.resize(img, (M_DIM, M_DIM))
print("Resized from ({}, {}, 3) to ({}, {}, 3)".format(old_shape[0], old_shape[1], M_DIM, M_DIM))

#Create the 2m x 2m checkerboard element by developing first row and second rows seperately, and then combining them via stacking
firstRow = np.concatenate((imgResized, 255 - imgResized), axis=1)
secondRow =  np.concatenate((255 - np.flip(imgResized, 0), np.flip(imgResized, 0)), axis=1)

checkerboard_element = np.concatenate((firstRow, secondRow), axis=0)

final = np.tile(checkerboard_element, (N_DIM, N_DIM, 1))

cv2.imwrite(OUTPUT_IMAGE, final)
print("The checkerboard with dimensions {} X {} was output to {}".format(2*N_DIM*M_DIM, 2*N_DIM*M_DIM, OUTPUT_IMAGE))
