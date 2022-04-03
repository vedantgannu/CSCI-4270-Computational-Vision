#!/usr/bin/env python
# coding: utf-8

import cv2
import sys
import os
import numpy as np


IMAGE_DIR = sys.argv[1]#"hw2_data_post/evergreen"
IMAGE_NAMES = sorted([file for file in os.listdir(IMAGE_DIR) if os.path.splitext(file)[1].lower() == ".jpg"])


info = []
for im in IMAGE_NAMES:
    img = cv2.imread(os.path.join(IMAGE_DIR, im), cv2.IMREAD_GRAYSCALE)
    im_dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    im_dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    squared_gradient = np.average(im_dx**2 + im_dy**2, axis=None)
    print("{}: {:.1f}".format(im, squared_gradient))
    info.append((im, squared_gradient))
print("Image {} is best focused.".format(sorted(info, key=lambda elem: elem[1])[-1][0]))
