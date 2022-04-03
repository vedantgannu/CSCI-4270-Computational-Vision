import numpy as np
import sys
import math

#python p1_camera.py params.txt points.txt
np.set_printoptions(precision=1, suppress=True)
PARAMS_FILE = sys.argv[1]#"hw2_data_post/params_p1_v3.txt"
POINTS_FILE = sys.argv[2]#"hw2_data_post/points_p1_v3.txt"


with open(PARAMS_FILE, mode='r') as params_fd:
    rotation_angles = np.array(params_fd.readline().strip("\n").split(), dtype='f')
    translation_vector = np.array(params_fd.readline().strip("\n").split(), dtype='f').reshape((3, 1))
    focal_length, pixel_d, vc, uc = np.array(params_fd.readline().strip("\n").split(), dtype='f')


rx_matrix = np.eye(3)
rx_matrix[1:3, 1:3] = np.array([[np.cos(rotation_angles[0] * (math.pi/180)), -1*np.sin(rotation_angles[0] * (math.pi/180))], [np.sin(rotation_angles[0] * (math.pi/180)), np.cos(rotation_angles[0] * (math.pi/180))]])
rz_matrix = np.eye(3)
rz_matrix[0:2, 0:2] = np.array([[np.cos(rotation_angles[2] * (math.pi/180)), -1*np.sin(rotation_angles[2] * (math.pi/180))], [np.sin(rotation_angles[2] * (math.pi/180)), np.cos(rotation_angles[2] * (math.pi/180))]])
ry_matrix = np.array([[np.cos(rotation_angles[1] * (math.pi/180)), 0, np.sin(rotation_angles[1] * (math.pi/180))], [0, 1, 0], [-1*np.sin(rotation_angles[1] * (math.pi/180)), 0, np.cos(rotation_angles[1] * (math.pi/180))]])
rotation_matrix = np.dot(np.dot(rx_matrix, ry_matrix), rz_matrix)


R_matrix = np.concatenate((rotation_matrix.T, np.dot(-1*rotation_matrix.T, translation_vector)), axis=1)

'''
for i in range(len(rotation_matrix)):
    print(*np.around(rotation_matrix[i], decimals=1), sep=", ")
'''

sx, sy = focal_length/(pixel_d*0.001), focal_length/(pixel_d*0.001)
K_matrix =  np.array([[sx, 0, uc], 
                      [0, sy, vc],
                      [0, 0, 1]])

'''
for i in range(len(K_matrix)):
    print(*np.around(K_matrix[i], decimals=1), sep=", ")
'''

M_matrix = np.dot(K_matrix, R_matrix)
print("Matrix M:")
for i in range(len(M_matrix)):
    print(*np.around(M_matrix[i], decimals=1), sep=", ")

print("Projections:")
with open(POINTS_FILE, mode='r') as points:
    visible = "visible:"
    hidden = "hidden:"
    for i, point in enumerate(points):
        point = point.strip("\n").split()
        homogeneous_points = np.array(point + [1], dtype='f').reshape((4,1))
        image_coords = np.dot(M_matrix, homogeneous_points).flatten()
        if image_coords[2] > 0:
            visible += " {}".format(i)
        else:
            hidden += " {}".format(i)
        affine_coordinates = np.around((image_coords / image_coords[2])[0:2][::-1], decimals=1).tolist()
        #Bounding box: 4000 rows, 6000 columns
        if affine_coordinates[0] >= 0 and affine_coordinates[0] < 4000 and affine_coordinates[1] >= 0 and affine_coordinates[1] < 6000:
            inside = 'inside'
        else:
            inside = 'outside'
        print("{}: {} => {} {}".format(i, " ".join(map(str, map(float, point))), " ".join(map(str, affine_coordinates)), inside))
    print(visible + "\n" + hidden)



