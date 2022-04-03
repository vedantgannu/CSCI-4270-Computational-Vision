import cv2
import numpy as np
import sys
import os


def seam_recursion(rowlocation, W_matrix, img_energy):
    if rowlocation == 0:#Base Case for bottom up approach: Assigning top row
        W_matrix[rowlocation, 1:W_matrix.shape[1]-1] = img_energy[rowlocation, 1:img_energy.shape[1]-1]
        return W_matrix
    else:
        W_matrix_modified = seam_recursion(rowlocation-1, W_matrix, img_energy)
        above_row = W_matrix_modified[rowlocation-1]
        left = above_row[:-2]#Get left view of row
        right = above_row[2:]#Get right view of row
        center = above_row[1:-1]#Get center view of row
        min_energy_neighbors = np.min(np.vstack((left, right, center)), axis=0)
        W_matrix_modified[rowlocation, 1:W_matrix_modified.shape[1]-1] = img_energy[rowlocation, 1:img_energy.shape[1]-1] + min_energy_neighbors
        return W_matrix_modified


np.set_printoptions(suppress=True)
IMG_PATH = sys.argv[1]
OUTPATH = "."
img = cv2.imread(IMG_PATH).astype(np.float32)
MODE = "vertical" if img.shape[0] < img.shape[1] else "horizontal"
ORIGINAL_SHAPE = img.shape
ORIGINAL_SHAPE_MOD = None
if MODE == "horizontal":#Transpose and the apply vertical seam removal to accomplish horizontal seam removal
    ORIGINAL_SHAPE_MOD = ORIGINAL_SHAPE
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#img = cv2.imread("hw3_data/autumn.jpg").astype(np.float32)
#img = cv2.imread("hw3_data/castle.jpg").astype(np.float32)
img_with_seam = img.copy()
ORIGINAL_SHAPE = img.shape
ITERATION = 0

while (img.shape[1] != img.shape[0]):
    img_energy = np.abs(cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0))\
                    + np.abs(cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1))
    W_matrix = np.empty((img.shape[0], img.shape[1]))
    W_matrix[:, [0,W_matrix.shape[1]-1]] = 10000000
    W_matrix = seam_recursion(W_matrix.shape[0]-1, W_matrix, img_energy)
    
    seam_coordinates = []
    starting_col = np.argmin(W_matrix[W_matrix.shape[0]-1])
    new_image = np.empty((img.shape[0], img.shape[1]-1, 3)) #Resized image will have one less column due to seam removal
    #Min Seam energy
    min_seam_energy = np.round(W_matrix[W_matrix.shape[0]-1, starting_col]/W_matrix.shape[0], decimals=2)
    seam_coordinates.append((W_matrix.shape[0]-1, starting_col))
    img_with_seam[-1, starting_col, :] = [0,0,250]
    image = []
    image.append(np.append(img[-1, 0:starting_col], img[-1, starting_col + 1:], axis=0))
    for i in range(W_matrix.shape[0]-2, -1, -1):#Trace back and get minimum seam pixels
        #Grab slice of the above row
        above_row_slice = W_matrix[i, [starting_col - 1, starting_col, starting_col + 1]]
        #Replace starting_col with above row's min val index
        column_index = starting_col - 1 + np.argmin(above_row_slice)
        seam_coordinates.append((i, column_index))
        #Edit the image for red seam
        img_with_seam[i, column_index, :] = [0,0,250]#Create the red seam
        image.append(np.append(img[i, 0:column_index], img[i, column_index + 1:], axis=0))#Create the new image
        starting_col = column_index
    if ITERATION in [0, 1, np.abs(ORIGINAL_SHAPE[1] - ORIGINAL_SHAPE[0] - 1)]:
        seam_coordinates = seam_coordinates[::-1]
        print("\nPoints on seam {}".format(ITERATION))
        print(MODE)
        if ITERATION == 0:
            ext = os.path.splitext(IMG_PATH)[1]
            target_file_path = os.path.join(OUTPATH, os.path.splitext(os.path.split(IMG_PATH)[1])[0] + "_seam" + ext)
            if MODE == "horizontal":
                cv2.imwrite(target_file_path, cv2.rotate(img_with_seam.astype("uint8"), cv2.ROTATE_90_COUNTERCLOCKWISE))
            else:
                cv2.imwrite(target_file_path, img_with_seam.astype("uint8"))
                
        if MODE == "horizontal":
            print("{}, {}".format(ORIGINAL_SHAPE_MOD[0] - seam_coordinates[0][1] - 1, seam_coordinates[0][0]))
            print("{}, {}".format(ORIGINAL_SHAPE_MOD[0] - seam_coordinates[W_matrix.shape[0]//2][1] - 1, seam_coordinates[W_matrix.shape[0]//2][0]))
            print("{}, {}".format(ORIGINAL_SHAPE_MOD[0] - seam_coordinates[W_matrix.shape[0]-1][1] - 1, seam_coordinates[W_matrix.shape[0]-1][0]))
            print("Energy of seam {}: {}\n".format(ITERATION, min_seam_energy))
        else: 
            print("{}, {}".format(seam_coordinates[0][0], seam_coordinates[0][1]))
            print("{}, {}".format(seam_coordinates[W_matrix.shape[0]//2][0], seam_coordinates[W_matrix.shape[0]//2][1]))
            print("{}, {}".format(seam_coordinates[W_matrix.shape[0]-1][0], seam_coordinates[W_matrix.shape[0]-1][1]))
            print("Energy of seam {}: {:.2f}".format(ITERATION, min_seam_energy))
        
    image = np.array(image)[::-1, :, :]#Reverse order of the rows
    img = image
    if MODE == "horizontal":
        ORIGINAL_SHAPE_MOD = (ORIGINAL_SHAPE_MOD[0] - 1, ORIGINAL_SHAPE_MOD[1])
    ITERATION += 1

target_file_path = os.path.join(OUTPATH, os.path.splitext(os.path.split(IMG_PATH)[1])[0] + "_final" + ext)
if MODE == "horizontal":
    cv2.imwrite(target_file_path, cv2.rotate(img.astype("uint8"), cv2.ROTATE_90_COUNTERCLOCKWISE))
else:
    cv2.imwrite(target_file_path, img.astype("uint8"))


'''




if (img.shape[0] < img.shape[1]):#Vertical seam case
    while (img.shape[1] != img.shape[0]):
        img_energy = np.abs(cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0))\
                        + np.abs(cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1))
        W_matrix = np.empty((img.shape[0], img.shape[1]))
        W_matrix[:, [0,W_matrix.shape[1]-1]] = 10000000
        W_matrix = seam_recursion(W_matrix.shape[0]-1, W_matrix, img_energy)
        
        seam_coordinates = []
        starting_col = np.argmin(W_matrix[W_matrix.shape[0]-1])
        new_image = np.empty((img.shape[0], img.shape[1]-1, 3)) #Resized image will have one less column due to seam removal
        #Min Seam energy
        min_seam_energy = np.round(W_matrix[W_matrix.shape[0]-1, starting_col]/W_matrix.shape[0], decimals=2)
        seam_coordinates.append((W_matrix.shape[0]-1, starting_col))
        img_with_seam[-1, starting_col, :] = [0,0,250]
        image = []
        image.append(np.append(img[-1, 0:starting_col], img[-1, starting_col + 1:], axis=0))
        for i in range(W_matrix.shape[0]-2, -1, -1):#Trace back and get minimum seam pixels
            #Grab slice of the above row
            above_row_slice = W_matrix[i, [starting_col - 1, starting_col, starting_col + 1]]
            #Replace starting_col with above row's min val index
            column_index = starting_col - 1 + np.argmin(above_row_slice)
            seam_coordinates.append((i, column_index))
            #Edit the image for red seam
            img_with_seam[i, column_index, :] = [0,0,250]#Create the red seam
            image.append(np.append(img[i, 0:column_index], img[i, column_index + 1:], axis=0))#Create the new image
            starting_col = column_index
        if ITERATION in [0, 1, ORIGINAL_SHAPE[1] - ORIGINAL_SHAPE[0] - 1]:
            seam_coordinates = seam_coordinates[::-1]
            print("\nPoints on seam {}".format(ITERATION))
            print("vertical")
            if ITERATION == 0:
                cv2.imshow("image", img_with_seam.astype("uint8"))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            print("{}, {}".format(seam_coordinates[0][0], seam_coordinates[0][1]))
            print("{}, {}".format(seam_coordinates[W_matrix.shape[0]//2][0], seam_coordinates[W_matrix.shape[0]//2][1]))
            print("{}, {}".format(seam_coordinates[W_matrix.shape[0]-1][0], seam_coordinates[W_matrix.shape[0]-1][1]))
            print("Energy of seam {}: {:.2f}".format(ITERATION, min_seam_energy))
            
        image = np.array(image)[::-1, :, :]#Reverse order of the rows
        img = image
        ITERATION += 1
else:#Horizontal seam case
    #Rotate image and convert it into a vertical seam removal case
    ORIGINAL_SHAPE = img.shape
    ORIGINAL_SHAPE_MOD = ORIGINAL_SHAPE
    print(ORIGINAL_SHAPE)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("image", img.astype("uint8"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_with_seam = img.copy()
    #ORIGINAL_SHAPE = img.shape
    
    
    while (img.shape[1] != img.shape[0]):
        img_energy = np.abs(cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0))\
                        + np.abs(cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1))
        W_matrix = np.empty((img.shape[0], img.shape[1]))
        W_matrix[:, [0,W_matrix.shape[1]-1]] = 10000000
        W_matrix = seam_recursion(W_matrix.shape[0]-1, W_matrix, img_energy)
        
        seam_coordinates = []
        starting_col = np.argmin(W_matrix[W_matrix.shape[0]-1])
        new_image = np.empty((img.shape[0], img.shape[1]-1, 3)) #Resized image will have one less column due to seam removal
        #Min Seam energy
        min_seam_energy = np.round(W_matrix[W_matrix.shape[0]-1, starting_col]/W_matrix.shape[0], decimals=2)
        seam_coordinates.append((W_matrix.shape[0]-1, starting_col))
        img_with_seam[-1, starting_col, :] = [0,0,250]
        image = []
        image.append(np.append(img[-1, 0:starting_col], img[-1, starting_col + 1:], axis=0))
        for i in range(W_matrix.shape[0]-2, -1, -1):#Trace back and get minimum seam pixels
            #Grab slice of the above row
            above_row_slice = W_matrix[i, [starting_col - 1, starting_col, starting_col + 1]]
            #Replace starting_col with above row's min val index
            column_index = starting_col - 1 + np.argmin(above_row_slice)
            seam_coordinates.append((i, column_index))
            #Edit the image for red seam
            img_with_seam[i, column_index, :] = [0,0,250]#Create the red seam
            image.append(np.append(img[i, 0:column_index], img[i, column_index + 1:], axis=0))#Create the new image
            starting_col = column_index
        if ITERATION in [0, 1, ORIGINAL_SHAPE[0] - ORIGINAL_SHAPE[1] - 1]:
            seam_coordinates = seam_coordinates[::-1]
            print("Points on seam {}".format(ITERATION))
            print("horizontal")
            if ITERATION == 0:
                cv2.imshow("image", cv2.rotate(img_with_seam.astype("uint8"), cv2.ROTATE_90_COUNTERCLOCKWISE))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            print("{}, {}".format(ORIGINAL_SHAPE_MOD[0] - seam_coordinates[0][1] - 1, seam_coordinates[0][0]))
            print("{}, {}".format(ORIGINAL_SHAPE_MOD[0] - seam_coordinates[W_matrix.shape[0]//2][1] - 1, seam_coordinates[W_matrix.shape[0]//2][0]))
            print("{}, {}".format(ORIGINAL_SHAPE_MOD[0] - seam_coordinates[W_matrix.shape[0]-1][1] - 1, seam_coordinates[W_matrix.shape[0]-1][0]))
            #print("{}, {}".format(seam_coordinates[W_matrix.shape[0]-1][0], seam_coordinates[W_matrix.shape[0]-1][1]))
            print("Energy of seam {}: {}\n".format(ITERATION, min_seam_energy))
            
        image = np.array(image)[::-1, :, :]#Reverse order of the rows
        img = image
        ORIGINAL_SHAPE_MOD = (ORIGINAL_SHAPE_MOD[0] - 1, ORIGINAL_SHAPE_MOD[1])
        ITERATION += 1

    cv2.imshow("image", cv2.rotate(img.astype("uint8"), cv2.ROTATE_90_COUNTERCLOCKWISE))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
'''