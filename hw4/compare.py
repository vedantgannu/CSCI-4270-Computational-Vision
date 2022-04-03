import cv2
import numpy as np
import sys
import os
from matplotlib import pyplot as plt


path1 = sys.argv[1]
path2 = sys.argv[2]
should_match = int(sys.argv[3])
orig_im1 = cv2.imread(path1)
orig_im2 = cv2.imread(path2)
im1 = cv2.cvtColor(orig_im1, cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(orig_im2, cv2.COLOR_BGR2GRAY)

print("Testing {} and {}".format(path1, path2))
if should_match == 1:
    print("Images are supposed to match")
else:
    print("Images are not supposed to match")

sift_alg = cv2.SIFT_create()
sift_kp1, sift_descriptors1 = sift_alg.detectAndCompute(im1.astype("uint8"),None)
sift_kp2, sift_descriptors2 = sift_alg.detectAndCompute(im2.astype("uint8"),None)

#Ratio Test
bf = cv2.BFMatcher()
matches = bf.knnMatch(sift_descriptors1,sift_descriptors2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.80*n.distance:
        good.append([m])

#Symmetric Matching
sym_match_im1_pts = []
sym_match_im2_pts = []

for i in range(len(sift_descriptors1)):
    #Get norm distances from image 1 descriptor 'i' to all descriptors in image 2
    norm_distances = np.linalg.norm(sift_descriptors2 - sift_descriptors1[i], axis=1)
    min_j_index = np.argmin(norm_distances, axis=0)
    norm_distances1 = np.linalg.norm(sift_descriptors1 - sift_descriptors2[min_j_index], axis=1)
    min_i_index = np.argmin(norm_distances1, axis=0)
    if i == min_i_index:
        sym_match_im1_pts.append(sift_kp1[i].pt)
        sym_match_im2_pts.append(sift_kp2[min_j_index].pt)

print("Number of matched keypoints using Ratio Test:", len(good))
print("Number of matched keypoints using Symmetric Matching Test:", len(sym_match_im1_pts))
print("Percentage of matched keypoints using Ratio Test:", len(good)/min(len(sift_kp1), len(sift_kp2)))
print("Percentage of matched keypoints using Symmetric Matching Test:", len(sym_match_im1_pts)/min(len(sift_kp1), len(sift_kp2)))
print()

if should_match == 1:
    inlier_im1_pts = [sift_kp1[match_ls[0].queryIdx].pt for match_ls in good]
    inlier_im2_pts = [sift_kp2[match_ls[0].trainIdx].pt for match_ls in good]
    fundamental_matrix, fundamental_mask = cv2.findFundamentalMat(np.array(inlier_im1_pts), np.array(inlier_im2_pts), cv2.FM_RANSAC, 5, 0.95, 500)
    inlier_fundamental_matches = np.array(good)[fundamental_mask.astype(bool)]
    print("Number of inlier matched keypoints after Ratio Test:", len(inlier_fundamental_matches))
    print("Percentage of inlier matched keypoints after Ratio Test:", len(inlier_fundamental_matches)/len(good)*100)


    inliers_symmetric = []
    for i in range(len(sym_match_im1_pts)):
        epilpolar_line_params = cv2.computeCorrespondEpilines(np.array(sym_match_im1_pts)[i].reshape(1,2), 1, fundamental_matrix).reshape(-1,3)
        distance = epilpolar_line_params @ np.array(list(sym_match_im2_pts[i]) + [1]).reshape(3, -1)
        if distance / np.linalg.norm(epilpolar_line_params[0][0:2]) <= 5:
            inliers_symmetric.append([sym_match_im1_pts[i], sym_match_im2_pts[i]])
    print("Number of inlier matched keypoints after Symmetric Matching:", len(inliers_symmetric))
    print("Percentage of inlier matched keypoints after Symmetric Matching:", len(inliers_symmetric)/len(sym_match_im1_pts)*100)
    print()



