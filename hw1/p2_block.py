#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
import sys
import os

if len(sys.argv) != 5:
    print("Missing CMD arguments")
    sys.exit(1)
FILENAME = sys.argv[1]
DOWNSIZE_ROWS = int(sys.argv[2])
DOWNSIZE_COLS = int(sys.argv[3])
BLOCKS = int(sys.argv[4])
img = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
scaleFactorRow = img.shape[0]/DOWNSIZE_ROWS
scaleFactorCol = img.shape[1]/DOWNSIZE_COLS
print("Downsized images are ({}, {})".format(DOWNSIZE_ROWS, DOWNSIZE_COLS))
print("Block images are ({}, {})".format(DOWNSIZE_ROWS*BLOCKS, DOWNSIZE_COLS*BLOCKS))

downsized_img = np.zeros((DOWNSIZE_ROWS, DOWNSIZE_COLS))
for i in range(DOWNSIZE_ROWS):
    for j in range(DOWNSIZE_COLS):
        downsized_img[i,j] = np.mean(img[round(i*scaleFactorRow):round((i+1)*scaleFactorRow), round(j*scaleFactorCol):round((j+1)*scaleFactorCol)])
print("Average intensity at ({}, {}) is {:.2f}".format(DOWNSIZE_ROWS//4, DOWNSIZE_COLS//4, downsized_img[DOWNSIZE_ROWS//4, DOWNSIZE_COLS//4]))
print("Average intensity at ({}, {}) is {:.2f}".format(DOWNSIZE_ROWS//4, 3*DOWNSIZE_COLS//4, downsized_img[DOWNSIZE_ROWS//4, 3*DOWNSIZE_COLS//4]))
print("Average intensity at ({}, {}) is {:.2f}".format(3*DOWNSIZE_ROWS//4, DOWNSIZE_COLS//4, downsized_img[3*DOWNSIZE_ROWS//4, DOWNSIZE_COLS//4]))
print("Average intensity at ({}, {}) is {:.2f}".format(3*DOWNSIZE_ROWS//4, 3*DOWNSIZE_COLS//4, downsized_img[3*DOWNSIZE_ROWS//4, 3*DOWNSIZE_COLS//4]))

threshold = np.median(downsized_img)
print("Binary threshold: {:.2f}".format(threshold))
binary_downsized_img = np.where(downsized_img < threshold, 0, 255)
downsized_img = np.round(downsized_img)

final_img_g = np.repeat(np.repeat(downsized_img, BLOCKS, axis=0), BLOCKS, axis=1).astype(img.dtype)
final_img_b = np.repeat(np.repeat(binary_downsized_img, BLOCKS, axis=0), BLOCKS, axis=1).astype(img.dtype)
cv2.imwrite(os.path.splitext(FILENAME)[0] + "_g.jpg", final_img_g)
print("Wrote image {}".format(os.path.splitext(FILENAME)[0] + "_g.jpg"))
cv2.imwrite(os.path.splitext(FILENAME)[0] + "_b.jpg", final_img_b)
print("Wrote image {}".format(os.path.splitext(FILENAME)[0] + "_b.jpg"))

#cv2.imshow("image", final_img_b)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

