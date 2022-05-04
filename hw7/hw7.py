# %%
import cv2
import numpy as np
import sys
from sklearn.cluster import AgglomerativeClustering
from scipy.linalg import null_space
from matplotlib import pyplot as plt

path1 = sys.argv[1]#"hw7data/000199_10.png"
path2 = sys.argv[2]#"hw7data/000199_11.png"
im1 = cv2.imread(path1)
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2 = cv2.imread(path2)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)


def findFOE(point1, point2, motion1, motion2):
    #Get the normal vector params
    a1, a2 = (-motion1[1])/np.linalg.norm(motion1), (-motion2[1])/np.linalg.norm(motion2)
    b1, b2 = (-motion1[0])/np.linalg.norm(motion1), (-motion2[0])/np.linalg.norm(motion2)
    
    c1, c2 = -(a1*point1[0] + b1*point1[1]), -(a2*point2[0] + b2*point2[1])
    mat = np.array(([a1,b1,c1],[a2,b2,c2]))

    # finding nullspace
    pqr = null_space(mat)
    pqr = pqr * np.sign(pqr[0,0])
    
    #Find xFOE = p/r and yFOE = q/r
    FOE = np.array([pqr[0]/pqr[2],pqr[1]/pqr[2]])
    return FOE

#Courtesy of  https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 6000,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#Find good corners on the first image
p0 = cv2.goodFeaturesToTrack(im1_gray, mask = None, **feature_params)

#Find the corresponding points to image 1
p1, st, err = cv2.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)
# Select good points
if p1 is not None:
    good_new = p1[st==1]
    good_old = p0[st==1]
print("Number of interest points for flow:", len(good_new))
#Normalized motion vectors
motion_vectors = (good_new - good_old)


ITERATIONS = len(good_new)
TAU = 110
#RANSAC:
num_inliers = 0
num_outliers = 0
best_inliers = None
best_outliers = None
best_outlier_motion_vectors = []
best_FOE = None
for i in range(ITERATIONS):
    index1 = np.random.randint(0, len(good_new), size=1)
    index2 = np.random.randint(0, len(good_new), size=1)
    if index1 == index2:
        continue
    point1, point2, motion1, motion2 = good_old[index1].flatten(), good_old[index2].flatten(), motion_vectors[index1].flatten(), motion_vectors[index2].flatten()
    FOE_point = findFOE(point1, point2, motion1, motion2)
    
    #Iterate over the rest of the points, create the lines, and determine if each line is an inlier or outlier relative to the FOE point
    num_inliers_temp = 0
    inliers = []
    outliers = []
    outlier_motion_vectors = []
    for index in range(len(good_old)):
        a = (-motion_vectors[index].flatten()[1])/np.linalg.norm(motion_vectors[index])
        b = (-motion_vectors[index].flatten()[0])/np.linalg.norm(motion_vectors[index])
        c = -(a*good_old[index].flatten()[0] + b*good_old[index].flatten()[1])
        
        if np.abs(a*FOE_point[0] + b*FOE_point[1] + c) < TAU:
            inliers.append(good_old[index].astype(int))
        else:
            outliers.append(good_old[index].astype(int))
            outlier_motion_vectors.append(motion_vectors[index].flatten())
    if num_inliers < len(inliers):
        num_inliers = len(inliers)
        num_outliers = len(outliers)
        best_inliers = inliers[:]
        best_outliers = outliers[:]
        best_outlier_motion_vectors = outlier_motion_vectors[:]
        best_FOE = FOE_point

print("FOE:", best_FOE)
print("Number of inliers:", num_inliers)
print("Number of outliers:", num_outliers)
im3 = im1.copy()#For drawing points and motion vectors over image 1

im3 = cv2.circle(im3, (int(best_FOE[0][0]), int(best_FOE[1][0])), 5, [0,0,255], -1)
if num_inliers/len(good_new) > 0.5:
    print("Detected camera motion")
    print("Drawing FOE")
else:
    print("No camera motion detected")

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    im3 = cv2.line(im3, (int(a), int(b)), (int(c), int(d)), [255,0,255], 2)
    im3 = cv2.circle(im3, (int(c), int(d)), 3, [0,140,255], -1)#points from image 1
    im3 = cv2.circle(im3, (int(a), int(b)), 3, [255,0,0], -1)#points from image 2

for inlier in best_inliers:
    a, b = inlier.ravel()
    im3 = cv2.circle(im3, (int(a), int(b)), 3, [0,255,0], -1)
cv2.imwrite('output_01.jpg', im3)
#plt.figure(figsize=(30,30))
#plt.imshow(cv2.cvtColor(im3, cv2.COLOR_BGR2RGB))
#Orange means outliers, blue means interest points in image 2, green means inlier points, and red is FOE


im4 = im1.copy()
print("Finding independent object motion clusters")
# k-mean clustering of outlier points (ind moving objects)
kmeans = AgglomerativeClustering(n_clusters = None, distance_threshold = 100).fit(best_outliers)
label_map = kmeans.labels_
print("Number of clusters:", len(set(label_map)))
#Remove bad clusters
freq = np.array(np.unique(label_map, return_counts=True))
good_cluster_indices = np.where(freq[1,:] > 4)
mask = np.isin(label_map, label_map[list(good_cluster_indices)])
new_label_map = label_map[mask]
print("New Number of Clusters:", len(set(new_label_map)))

#Draw clusters
colors = np.random.randint(0, 255, (max(label_map) + 1, 3))
best_outliers_np = np.array(best_outliers)[mask]
best_outlier_motion_vectors_np = np.array(best_outlier_motion_vectors)[mask]

for cluster_group in set(new_label_map):
    #Draw bounding box
    indices = np.where(new_label_map == cluster_group)
    min_x, min_y = np.min(best_outliers_np[list(indices)], axis=0).ravel()
    max_x, max_y = np.max(best_outliers_np[list(indices)], axis=0).ravel()
    im4 = cv2.rectangle(im4, (min_x-6,min_y-6),(max_x+6,max_y+6),colors[cluster_group].tolist(),2)
    #Draw in the points and motion vectors
    for point, motion_vector in zip(best_outliers_np[list(indices)], best_outlier_motion_vectors_np[list(indices)]):
        a, b = point.ravel()
        c, d = (point + motion_vector).ravel()
        im4 = cv2.line(im4, (int(a), int(b)), (int(c), int(d)), colors[cluster_group].tolist(), 2)
        im4 = cv2.circle(im4, (int(a), int(b)), 3, colors[cluster_group].tolist(), -1)#points from image 1
        im4 = cv2.circle(im4, (int(c), int(d)), 3, colors[cluster_group].tolist(), -1)#points from image 2

cv2.imwrite('output_02.jpg', im4)
#plt.figure(figsize=(30,30))
#plt.imshow(cv2.cvtColor(im4, cv2.COLOR_BGR2RGB))
    
