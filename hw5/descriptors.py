#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.utils.random import sample_without_replacement

if __name__ == "__main__":
    root_folders = ["train", "valid", "test"]

    for root_folder in root_folders:
        image_folders = os.listdir(root_folder)
        for image_folder in image_folders:
            descriptor_matrix = []
            current_path = os.path.join(root_folder, image_folder)
            print("Getting descriptors of", current_path)
            for image_file in os.listdir(os.path.join(root_folder, image_folder)):#Turn each image into a descriptor vector
                img = cv2.imread(os.path.join(root_folder, image_folder, image_file))
                ROWS, COLUMNS = img.shape[0:2]
                num_subimage_blocks_width = 4
                num_subimage_blocks_height = 4
                subimage_block_width = int(COLUMNS / (num_subimage_blocks_width + 1))
                subimage_block_height = int(ROWS / (num_subimage_blocks_height + 1))
                result_vector = []
                histogram_bins = 4
                for i in range(num_subimage_blocks_height):
                    for j in range(num_subimage_blocks_width):
                        subimage_block = img[i*subimage_block_height: (i + 2)*subimage_block_height, j*subimage_block_width: (j + 2)*subimage_block_width,:]
                        hist, _ = np.histogramdd(subimage_block.reshape(-1,3), (histogram_bins, histogram_bins, histogram_bins))
                        result_vector += hist.flatten().tolist()
                descriptor_matrix.append(result_vector)
            export_file_name = "{}_{}.npz".format(root_folder, image_folder)
            export_file_path = os.path.join(root_folder, export_file_name)
            print("Fetched all descriptors. Exporting to", export_file_path)
            np.savez_compressed(export_file_path, np.array(descriptor_matrix))
