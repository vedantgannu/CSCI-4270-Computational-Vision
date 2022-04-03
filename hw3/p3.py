import cv2         
import numpy as np
from matplotlib import pyplot as plt
import os
import sys


filename,ext = os.path.splitext(sys.argv[1])

output_path = sys.argv[3]

im0 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE).astype(float)
im1 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE).astype(float)
final = cv2.imread(sys.argv[1]).astype(float)


im_diff = np.abs(im0 - im1)
'''
plt.gray()
plt.imshow(im_diff)
plt.show()
'''
max_diff = np.max(im_diff)
mean = np.average(im_diff)
std = np.std(im_diff)
thresh = mean + 2*std



#Black out the foreground seed holder since seed distortion shouldn't be considered, and everything on the left half of the image
im_diff1 = im_diff.copy()
im_diff1[im1.shape[0]*3//4:im1.shape[0], im1.shape[1]//2:im1.shape[1]] = 0
im_diff1[:im1.shape[0], 0:im1.shape[1]//2] = 0
'''
plt.imshow(im_diff1.astype("uint8"))
plt.show()
'''

#Get rid of small salt and pepper noise
im_diff1 = cv2.medianBlur(im_diff1.astype("uint8"), ksize=7)
retval,im_thresh = cv2.threshold(im_diff1.astype(np.uint8),thresh,255,cv2.THRESH_BINARY)

kernel = np.ones((12,12),np.uint8)

after_close = cv2.morphologyEx(im_thresh,cv2.MORPH_CLOSE,kernel)
im_close_first = cv2.morphologyEx(after_close,cv2.MORPH_OPEN,kernel)

im_close_first = cv2.morphologyEx(im_close_first,cv2.MORPH_OPEN,np.ones((15,15),np.uint8))


'''
The next step is connected components labeling
'''
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im_close_first, connectivity, cv2.CV_32S)


colors = np.random.randint(255, size=(num_labels,3))
colors[0] = np.array((0,0,0))

pixel_ids, pixel_counts = np.unique(labels, return_counts=True)

if len(pixel_ids) == 1:
    #print("No components left")
    final[np.where(im_diff != 0)] = [100,100,100]
    cv2.imwrite(output_path, final.astype("uint8"))
    '''
    plt.imshow(final)
    plt.show()
    '''
    print("NO")
    sys.exit(0)


im_colored = colors[labels]


lower_right_center_coords = np.tile(np.array([im_colored.shape[1]*3//4, im_colored.shape[0]*3//4]), (len(centroids), 1))
optical_center = np.tile(np.array([im_colored.shape[1]//2, im_colored.shape[0]//2]), (len(centroids), 1))

#print(np.square(np.sum(centroids[1:,:] - lower_right_center_coords[1:,:], axis=1)))
distances = np.linalg.norm(centroids[1:,:] - lower_right_center_coords[1:,:], axis=1)
to_right = np.sum(distances)
distances = np.linalg.norm(centroids[1:,:] - optical_center[1:,:], axis=1)
to_center = np.sum(distances)

if to_right < to_center:
    final[np.where(im_diff != 0)] = [100,100,100]
    final[np.where(labels != 0)] = [185, 255, 218]
    cv2.imwrite(output_path, final.astype("uint8"))
    '''
    plt.imshow(final)
    plt.show()
    '''
    print("YES")
    sys.exit(0)
    


if to_right > to_center:
    print("Initial analysis failed, filtering out components that have less than 4000 pixels")
    pixel_ids, pixel_counts = np.unique(labels, return_counts=True)
    
    ls = np.argwhere(pixel_counts > 4000).flatten().tolist()
    
    if len(ls) == 1:
        #print("No components left")
        final[np.where(im_diff != 0)] = [100,100,100]
        cv2.imwrite(output_path, final.astype("uint8"))
        print("NO")
        '''
        plt.imshow(final)
        plt.show()
        '''
        sys.exit(0)

    for i in list(set(pixel_ids) - set(ls)):
        labels[np.where(labels == i)] = 0

    im_colored1 = colors[labels]


    new_slice = np.array(list(set(ls) - set([0])), dtype="int")
    distances = np.linalg.norm(centroids[new_slice,:] - lower_right_center_coords[new_slice,:], axis=1)
    
    to_right = np.sum(distances)
    distances = np.linalg.norm(centroids[new_slice,:] - optical_center[new_slice,:], axis=1)
    to_center = np.sum(distances)

    if to_right < to_center:
        final[np.where(im_diff != 0)] = [100,100,100]
        final[np.where(labels != 0)] = [185, 255, 218]
        '''
        plt.imshow(final)
        plt.show()
        '''
        cv2.imwrite(output_path, final.astype("uint8"))
        print("YES")
        sys.exit(0)
    

    final[np.where(im_diff != 0)] = [100,100,100]
    cv2.imwrite(output_path, final.astype("uint8"))
    
    '''
    plt.imshow(final)
    plt.show()
    '''
    print("NO")
