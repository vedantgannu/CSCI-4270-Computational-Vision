#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
import sys


if len(sys.argv) != 4:
    print("Missing CMD arguments")
    sys.exit(1)
elif sys.argv[3] not in ["left", "top", "right", "bottom", "center"]:
    print("dir argument ({}) not valid".format(sys.argv[3]))
    sys.exit(1)


IN_IMG = sys.argv[1]
OUT_DIR = sys.argv[2]
DIR = sys.argv[3].lower()
img = cv2.imread(IN_IMG)

M_rows = img.shape[0]
N_columns = img.shape[1]

shaded_img = None
multiplier_matrix = None
if DIR == "right":
    multiplier_matrix = (np.tile(np.arange(N_columns), (M_rows, 1))/(N_columns-1)) #(row, column) multiplers
elif DIR == "left":
    multiplier_matrix = (np.tile(np.arange(N_columns)[::-1], (M_rows, 1))/(N_columns-1)) #(row, column) multiplers
elif DIR == "bottom":
    multiplier_matrix = np.tile(np.arange(M_rows), (N_columns, 1)).T/(M_rows-1) #(row, column) multiplers
elif DIR == "top":
    multiplier_matrix = np.tile(np.arange(M_rows)[::-1], (N_columns, 1)).T/(M_rows-1) #(row, column) multiplers
elif DIR == "center":
    #Assembling the distance matrices by combining the left and right haves with the middle row/column
    row_dist_matrix = np.concatenate((np.tile(np.arange(M_rows//2 + 1)[1:][::-1], (N_columns,1)).T, np.zeros((1, N_columns)), np.tile(np.arange(M_rows - M_rows//2)[1:], (N_columns,1)).T), axis=0)
    col_dist_matrix = np.concatenate((np.tile(np.arange(N_columns//2 + 1)[1:][::-1], (M_rows,1)), np.zeros((M_rows, 1)), np.tile(np.arange(N_columns - N_columns//2)[1:], (M_rows,1))), axis=1)
    #Create Euclidean distance matrix from center pixel (M//2, N//2)
    combined_distance_matrix = np.sqrt(np.square(col_dist_matrix) + np.square(row_dist_matrix))
    #Normalize
    multiplier_matrix = np.abs((combined_distance_matrix - np.max(combined_distance_matrix)) / np.max(combined_distance_matrix))

for row in [0, M_rows//2, M_rows-1]:
    for column in [0, N_columns//2, N_columns-1]:
        print("({},{}) {:.3f}".format(row, column, multiplier_matrix[row,column]))
shaded_img = (img * multiplier_matrix.reshape(M_rows, N_columns, 1)).astype(img.dtype)
final_img = np.concatenate((img, shaded_img), axis=1)


cv2.imwrite(OUT_DIR, final_img)
#cv2.imshow("image", final_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


