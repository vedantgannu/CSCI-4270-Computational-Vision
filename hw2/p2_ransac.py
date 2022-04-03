#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys

np.set_printoptions(precision=3)
POINTS = sys.argv[1]#"hw2_data_post/p2_pts1_in.txt"
SAMPLES = int(sys.argv[2])#25
TAU = float(sys.argv[3])#2.5
SEED = None#999
if len(sys.argv) == 5:
   SEED = int(sys.argv[4])

points_matrix = None
#Read points into matrix
list1 = []
with open(POINTS, mode="r") as points_fd:
    for point in points_fd:
        list1.append(np.array(point.strip("\n").split(), dtype='f'))
    points_matrix = np.array(list1)

np.random.seed(SEED)
#Using implicit definition of a line: 0 = ax + by + c
kmax = 0
best_inlier_distances = []
best_outlier_distances = []
for i in range(SAMPLES):
    index1, index2 = np.random.randint(0, len(points_matrix), 2)
    if index1 == index2:
        continue
    #(y1-y2)
    a= points_matrix[index1][1] - points_matrix[index2][1]
    #(x2-x1)
    b= points_matrix[index2][0] - points_matrix[index1][0]
    #c=#(y2(x1-x2) + x2(y2-y1))
    intercept = points_matrix[index2][1]*points_matrix[index1][0] - points_matrix[index2][0]*points_matrix[index1][1]
    a1 = a / np.sqrt(np.sum(np.square([a, b])))
    b1 = b / np.sqrt(np.sum(np.square([a, b])))
    intercept1 = intercept / np.sqrt(np.sum(np.square([a, b])))
    if intercept1 > 0:
        a1 *= -1
        b1 *= -1
        intercept1 *= -1
    inlier_distances = []
    outlier_distances = []
    for index in np.arange(len(points_matrix)):
        if np.abs(a1*points_matrix[index][0] + b1*points_matrix[index][1] + intercept1) < TAU:
            inlier_distances.append(np.abs(a1*points_matrix[index][0] + b1*points_matrix[index][1] + intercept1))
        else:
            outlier_distances.append(np.abs(a1*points_matrix[index][0] + b1*points_matrix[index][1] + intercept1))
    if len(inlier_distances) > kmax:
        print("Sample {}:".format(i))
        print("indices ({},{})".format(index1, index2))
        print("line ({:.3f},{:.3f},{:.3f})".format(a1, b1, intercept1))
        print("inliers {}\n".format(len(inlier_distances)))
        kmax = len(inlier_distances)
        best_inlier_distances = inlier_distances
        best_outlier_distances = outlier_distances
print("avg inlier dist {:.3f}".format(np.mean(np.array(best_inlier_distances))))
print("avg outlier dist {:.3f}".format(np.mean(np.array(best_outlier_distances))))

