import cv2
import numpy as np
import sys
import warnings
import os

warnings.filterwarnings('ignore')


IMG = sys.argv[2]#"hw3_data/disk.png"#"hw3_data/tree_sky.jpg"
file, ext = os.path.splitext(IMG)

img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)

sigma = float(sys.argv[1])#1.0#1.5
width = max(3, int(4*sigma+1))
ksize = (width,width)
im_s = cv2.GaussianBlur(img.astype(np.float32), ksize, sigma)

kx,ky = cv2.getDerivKernels(1,1,3)
kx = np.transpose(kx/2)
ky = ky/2
im_dx = cv2.filter2D(im_s, -1, kx)
im_dy = cv2.filter2D(im_s, -1, ky)

gradient_magnitudes = np.sqrt(im_dx**2 + im_dy**2)
gradient_magnitudes_output = (gradient_magnitudes / np.max(gradient_magnitudes) * 255).astype("uint8")

cv2.imwrite(file + "_grd" + ext, gradient_magnitudes_output)

gradient_directions = np.arctan2(im_dy, im_dx)
gradient_directions[gradient_directions != 0] *= 180/np.pi


edgels = np.zeros(gradient_magnitudes.shape)
#Determining edgels from pixels with East or West facing gradient orientation
#Getting coordinates of pixels with gradient orientation East or West
east_west_coordinates = np.where(
                                ((gradient_directions > -22.5) & (gradient_directions < 22.5))
                                | ((gradient_directions < -157.5) | (gradient_directions > 157.5)))

temp_east_west_coordinates = np.array(east_west_coordinates)


behind_coordinates = (east_west_coordinates[0], east_west_coordinates[1] - 1)
ahead_coordinates = (east_west_coordinates[0], east_west_coordinates[1] + 1)

temp_behind = np.array(behind_coordinates)
cols_delete = []
cols_delete += np.where((temp_behind < 0))[1].tolist()#Getting columns that have negative index val for column

temp_ahead = np.array(ahead_coordinates)
cols_delete += np.where((temp_ahead == gradient_directions.shape[1]))[1].tolist()#Getting columns that have index val above column numbers
cols_delete = list(set(cols_delete))

temp_behind = np.delete(temp_behind, cols_delete, axis=1)
temp_ahead = np.delete(temp_ahead, cols_delete, axis=1)
temp_east_west_coordinates = np.delete(temp_east_west_coordinates, cols_delete, axis=1)


#Surpressing coordinates based on neighbors gradient mags

bool_arr = (gradient_magnitudes[list(temp_east_west_coordinates)] >= gradient_magnitudes[list(temp_behind)])\
    & (gradient_magnitudes[list(temp_east_west_coordinates)] >= gradient_magnitudes[list(temp_ahead)])\
    & (gradient_magnitudes[list(temp_east_west_coordinates)] != 0) & (gradient_magnitudes[list(temp_ahead)] != 0)\
    & (gradient_magnitudes[list(temp_behind)] != 0)

east_west_coordinates_maximum = np.delete(temp_east_west_coordinates, np.argwhere(bool_arr == False).flatten().tolist(), axis=1)
edgels[list(east_west_coordinates_maximum)] = 1 #Indicate edgels (red)

#North/South
north_south_coordinates = np.where(
                                ((gradient_directions > 67.5) & (gradient_directions < 112.5))
                                | ((gradient_directions > -112.5) & (gradient_directions < -67.5)))

temp_north_south_coordinates = np.array(north_south_coordinates)
behind_coordinates = (north_south_coordinates[0] - 1, north_south_coordinates[1])
ahead_coordinates = (north_south_coordinates[0] + 1, north_south_coordinates[1])

temp_behind = np.array(behind_coordinates)
cols_delete = []
cols_delete += np.where((temp_behind < 0))[1].tolist()#Getting columns that have negative index val for row (outside of image)

temp_ahead = np.array(ahead_coordinates)
cols_delete += np.where((temp_ahead == gradient_directions.shape[0]))[1].tolist()#Getting columns that have index val above row numbers (outside of image)
cols_delete = list(set(cols_delete))

temp_behind = np.delete(temp_behind, cols_delete, axis=1)
temp_ahead = np.delete(temp_ahead, cols_delete, axis=1)
temp_north_south_coordinates = np.delete(temp_north_south_coordinates, cols_delete, axis=1)

#Surpressing coordinates based on neighbors gradient mags. Handles the zero gradient issue by ensuring
#that for each pixel, the pixel and its neighbors have nonzero gradients

bool_arr = (gradient_magnitudes[list(temp_north_south_coordinates)] >= gradient_magnitudes[list(temp_behind)])\
    & (gradient_magnitudes[list(temp_north_south_coordinates)] >= gradient_magnitudes[list(temp_ahead)])\
    & (gradient_magnitudes[list(temp_north_south_coordinates)] != 0) & (gradient_magnitudes[list(temp_ahead)] != 0)\
    & (gradient_magnitudes[list(temp_behind)] != 0)


north_south_coordinates_maximum = np.delete(temp_north_south_coordinates, np.argwhere(bool_arr == False).flatten().tolist(), axis=1)
edgels[list(north_south_coordinates_maximum)] = 3 #Indicate edgels (blue)


#Northwest/Southeast
northwest_southeast_coordinates = np.where(
                                ((gradient_directions > 22.5) & (gradient_directions < 67.5))
                                | ((gradient_directions < -112.5) & (gradient_directions > -157.5)))

temp_northwest_southeast_coordinates = np.array(northwest_southeast_coordinates)
behind_coordinates = (northwest_southeast_coordinates[0] - 1, northwest_southeast_coordinates[1] - 1)
ahead_coordinates = (northwest_southeast_coordinates[0] + 1, northwest_southeast_coordinates[1] + 1)

temp_behind = np.array(behind_coordinates)
cols_delete = []
cols_delete += np.where((temp_behind < 0))[1].tolist()#Getting columns that have negative index val for row (outside of image)

temp_ahead = np.array(ahead_coordinates)
cols_delete += np.where((temp_ahead == gradient_directions.shape[0]) | (temp_ahead == gradient_directions.shape[1]))[1].tolist()#Getting columns that have index val above row numbers (outside of image)
cols_delete = list(set(cols_delete))

temp_behind = np.delete(temp_behind, cols_delete, axis=1)
temp_ahead = np.delete(temp_ahead, cols_delete, axis=1)
temp_northwest_southeast_coordinates = np.delete(temp_northwest_southeast_coordinates, cols_delete, axis=1)


#Surpressing coordinates based on neighbors gradient mags. Handles the zero gradient issue by ensuring
#that for each pixel, the pixel and its neighbors have nonzero gradients

bool_arr = (gradient_magnitudes[list(temp_northwest_southeast_coordinates)] >= gradient_magnitudes[list(temp_behind)])\
    & (gradient_magnitudes[list(temp_northwest_southeast_coordinates)] >= gradient_magnitudes[list(temp_ahead)])\
    & (gradient_magnitudes[list(temp_northwest_southeast_coordinates)] != 0) & (gradient_magnitudes[list(temp_ahead)] != 0)\
    & (gradient_magnitudes[list(temp_behind)] != 0)


northwest_southeast_coordinates_maximum = np.delete(temp_northwest_southeast_coordinates, np.argwhere(bool_arr == False).flatten().tolist(), axis=1)
edgels[list(northwest_southeast_coordinates_maximum)] = 2 #Indicate edgels (green)


#Northeast/Southwest
northeast_southwest_coordinates = np.where(
                                ((gradient_directions > 112.5) & (gradient_directions < 157.5))
                                | ((gradient_directions < -22.5) & (gradient_directions > -67.5)))

temp_northeast_southwest_coordinates = np.array(northeast_southwest_coordinates)
behind_coordinates = (northeast_southwest_coordinates[0] - 1, northeast_southwest_coordinates[1] + 1)
ahead_coordinates = (northeast_southwest_coordinates[0] + 1, northeast_southwest_coordinates[1] - 1)

temp_behind = np.array(behind_coordinates)
cols_delete = []
cols_delete += np.where((temp_behind < 0) | (temp_behind == gradient_directions.shape[1]))[1].tolist()#Getting columns that have negative index val for row (outside of image)

temp_ahead = np.array(ahead_coordinates)
cols_delete += np.where((temp_ahead == gradient_directions.shape[0]) | (temp_ahead < 0))[1].tolist()#Getting columns that have index val above row numbers (outside of image)
cols_delete = list(set(cols_delete))

temp_behind = np.delete(temp_behind, cols_delete, axis=1)
temp_ahead = np.delete(temp_ahead, cols_delete, axis=1)
temp_northeast_southwest_coordinates = np.delete(temp_northeast_southwest_coordinates, cols_delete, axis=1)

#Surpressing coordinates based on neighbors gradient mags. Handles the zero gradient issue by ensuring
#that for each pixel, the pixel and its neighbors have nonzero gradients

bool_arr = (gradient_magnitudes[list(temp_northeast_southwest_coordinates)] >= gradient_magnitudes[list(temp_behind)])\
    & (gradient_magnitudes[list(temp_northeast_southwest_coordinates)] >= gradient_magnitudes[list(temp_ahead)])\
    & (gradient_magnitudes[list(temp_northeast_southwest_coordinates)] != 0) & (gradient_magnitudes[list(temp_ahead)] != 0)\
    & (gradient_magnitudes[list(temp_behind)] != 0)

northeast_southwest_coordinates_maximum = np.delete(temp_northeast_southwest_coordinates, np.argwhere(bool_arr == False).flatten().tolist(), axis=1)
edgels[list(northeast_southwest_coordinates_maximum)] = 4 #Indicate edgels (white)


#Producing gradient direction image
gradient_direction_image = cv2.imread(IMG)
gradient_direction_image[east_west_coordinates] = [0, 0, 255]
gradient_direction_image[northwest_southeast_coordinates] = [0, 255, 0]
gradient_direction_image[north_south_coordinates] = [255, 0, 0]
gradient_direction_image[northeast_southwest_coordinates] = [255, 255, 255]
gradient_direction_image[np.where(gradient_magnitudes < 1.0)] = [0, 0, 0]
gradient_direction_image[:, [0, gradient_direction_image.shape[1]-1]] = [0, 0, 0]
gradient_direction_image[[0, gradient_direction_image.shape[0]-1], :] = [0, 0, 0]


cv2.imwrite(file + "_dir" + ext, gradient_direction_image.astype("uint8"))


#Thresholding
edge_coordinates = np.where(edgels != 0)
print("Number after non-maximum:", np.array(edge_coordinates).shape[1])
bool_arr = gradient_magnitudes[edge_coordinates] < 1.0
edge_coordinates_first_threshold_surpress = np.delete(np.array(edge_coordinates), np.argwhere(bool_arr == False).flatten().tolist(), axis=1)#Got rid of edgels that have grad < 1.0
edge_coordinates_first_threshold_remain = np.delete(np.array(edge_coordinates), np.argwhere(bool_arr == True).flatten().tolist(), axis=1)#Remaining edgels that have grad > 1.0


print("Number after 1.0 threshold:", edge_coordinates_first_threshold_remain.shape[1])
grad_threshold = gradient_magnitudes.copy()
grad_threshold[list(edge_coordinates_first_threshold_surpress)] = 0

mean = np.mean(gradient_magnitudes[list(edge_coordinates_first_threshold_remain)])
print("mu: {:.2f}".format(mean))
standard_deviation = np.std(gradient_magnitudes[list(edge_coordinates_first_threshold_remain)])
print("s: {:.2f}".format(standard_deviation))
threshold = min(mean + 0.5*standard_deviation, 30/sigma)
print("Threshold: {:.2f}".format(threshold))


bool_arr = gradient_magnitudes[list(edge_coordinates_first_threshold_remain)] < threshold
#Edgels coordinates that remain after pruning with the second threshold 
edge_coordinates_second_threshold_remain = np.delete(edge_coordinates_first_threshold_remain, np.argwhere(bool_arr == True).flatten().tolist(), axis=1)#Got rid of edgels that have grad < 1.0
#Edgels coordinates that are prunned using second threshold
edge_coordinates_second_threshold_surpress = np.delete(edge_coordinates_first_threshold_remain, np.argwhere(bool_arr == False).flatten().tolist(), axis=1)#Got rid of edgels that have grad < 1.0

print("Number after threshold: {}".format(edge_coordinates_second_threshold_remain.shape[1]))
grad_threshold[list(edge_coordinates_second_threshold_surpress)] = 0

cv2.imwrite(file + "_thr" + ext, (grad_threshold / np.max(grad_threshold) * 255).astype("uint8"))




