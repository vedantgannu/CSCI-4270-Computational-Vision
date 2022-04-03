#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
import os
import sys


IMG_DIR = sys.argv[1]
N = int(sys.argv[2])#Row and column dimensions of each region
IMG_NAMES = sorted([file for file in os.listdir(IMG_DIR) if os.path.splitext(file)[1] == '.jpg'])
IMG_DATA = []

#Iterate over each image name in sorted order
for file in IMG_NAMES:
    img = cv2.imread(os.path.join(IMG_DIR, file))
    scaleFactorRow = img.shape[0]/N
    scaleFactorCol = img.shape[1]/N
    downsized_img = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(N):
            #Slice original image regions, find their average BGR intensities, and associate them with appropriate downsized image pixel
            img_region = (img[round(i*scaleFactorRow):round((i+1)*scaleFactorRow), round(j*scaleFactorCol):round((j+1)*scaleFactorCol)]).reshape((-1,3))
            downsized_img[i,j] = np.mean(img_region, axis=0)
    IMG_DATA.append((file, downsized_img[:,:,::-1]))
print("Nearest distances")
first_image_normalized = IMG_DATA[0][1] / np.linalg.norm(IMG_DATA[0][1].flatten()) * 100
print("First region: {:.3f} {:.3f} {:.3f}".format(first_image_normalized[0,0,0], first_image_normalized[0,0,1], first_image_normalized[0,0,2]))
print("Last region: {:.3f} {:.3f} {:.3f}".format(first_image_normalized[N-1,N-1,0], first_image_normalized[N-1,N-1,1], first_image_normalized[N-1,N-1,2]))

#Compute descriptor vector pairs between each file and choose the most similar pair for each file
for file, downsized_img in IMG_DATA:
    descriptor_vector = downsized_img.flatten() / np.linalg.norm(downsized_img.flatten()) * 100
    pairs = []
    for file1, downsized_img1 in IMG_DATA:
        if file == file1:
            continue
        else:
            descriptor_vector1 = downsized_img1.flatten() / np.linalg.norm(downsized_img1.flatten()) * 100
            pairs.append((file1, np.linalg.norm(descriptor_vector - descriptor_vector1)))
    
    most_similar = sorted(pairs, key=lambda x : x[1])[0]
    print("{} to {}: {:.2f}".format(file, most_similar[0], most_similar[1]))

