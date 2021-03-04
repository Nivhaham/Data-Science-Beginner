#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np
from collections import Counter
import struct
from array import array
from os.path import join
import os
import random
import matplotlib.pyplot as plt
import random
from Ass2 import loadMNIST
import sys
# import Integer
import math

normalize = 255
k = 10
cell_size = 28
picture_centroid_idx = [-1] * 60000
picture_centroid_idx = np.array(picture_centroid_idx)


def run():
    #
    # Load MINST dataset
    #
    mnist_dataloader = loadMNIST.MnistDataloader(loadMNIST.training_images_filepath, loadMNIST.training_labels_filepath,
                                                 loadMNIST.test_images_filepath, loadMNIST.test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    x_train_np = np.array(x_train)
    x_train_np = x_train_np.reshape(60000, 28 * 28)
    x_train_np_normalize = x_train_np / normalize

    y_train_np = np.array(y_train)
    # print(y_train_np)

    centroids = np.random.randint(normalize, size=(k, 28, 28))
    centroids = centroids / normalize
    centroids_np = np.array(centroids)

    centroids_np = centroids_np.reshape(k, 28 * 28)
    map_image_to_centroids = np.zeros(60000)
    succuess = 0
    centroids_not_random = not_a_random_init(np.zeros((k, 28 * 28)), x_train_np_normalize)

    # print(y_train_np[1],y_train_np[3],y_train_np[5],y_train_np[7],y_train_np[2],y_train_np[0],y_train_np[13],y_train_np[15],y_train_np[17],y_train_np[4])
    # print(map_centroid_to_digit)
    # train
    for epoch in range(20):
        print("iteration number " + str(epoch + 1))
        assign_images_to_centroids(picture_centroid_idx, centroids_np, x_train_np_normalize)
        centroids_np = get_new_centroids(picture_centroid_idx, x_train_np_normalize)

        # not a random check
        #assign_images_to_centroids(picture_centroid_idx, centroids_not_random, x_train_np_normalize)
        #centroids_not_random = get_new_centroids(picture_centroid_idx, x_train_np_normalize)

    # now assign each image to centroid
    for i in range(60000):
         map_image_to_centroids[i] = get_closest_centroid(centroids_np, x_train_np_normalize[i])
         #not a random check
        #map_image_to_centroids[i] = get_closest_centroid(centroids_not_random, x_train_np_normalize[i])
    # print(map_image_to_centroids)
    map_centroid_to_digit = assign_label_to_centroids(y_train_np, picture_centroid_idx)
    print(map_centroid_to_digit)
    for i in range(60000):
        if (map_centroid_to_digit[picture_centroid_idx[i]] == y_train_np[i]):
            succuess += 1
    print("The succuess rate of the algorithem is  " + str(succuess / 60000 * 100) + "%")


# centroid to digit
# cluster_np is a numpyarray of all the images of the centroid.
def assign_label_to_centroids(y_train_np, picture_centroid_idx, ):
    counters = np.zeros((10, 10))
    map_centroid_to_digit = np.zeros(10)
    for i in range(60000):
        counters[picture_centroid_idx[i]][y_train_np[i]] += 1
        # print(counters)
    for i in range(10):
        # map_centroid_to_digit[i] = np.where( counters[i] == counters[i].max(),counters[i])
        map_centroid_to_digit[i] = np.argmax(counters[i])
    return map_centroid_to_digit


def get_closest_centroid(centroids, image):
    closest_centroid = -1
    min_dis = sys.maxsize

    for j in range(k):
        diff = np.linalg.norm(centroids[j] - image)
        if diff < min_dis:
            min_dis = diff
            closest_centroid = j
    # print(diff)
    # print(closest_centroid)
    return closest_centroid


def assign_images_to_centroids(picture_centroid_idx, centroieds_np, x_train_np_normalize):
    for i in range(60000):
        # check the distance of sample with all the centers.
        picture_centroid_idx[i] = get_closest_centroid(centroieds_np, x_train_np_normalize[i])


# which sample to which cluster index
def get_new_centroids(picture_centroid_idx, images):
    sum_images_per_centroid = np.zeros((10, 28 * 28))
    avg = np.zeros((10, 28 * 28))
    counter = np.zeros(10)
    for i in range(60000):
        sum_images_per_centroid[picture_centroid_idx[i]] += images[i]
        counter[picture_centroid_idx[i]] += 1
    count_idx = 0
    for i in range(10):
        if counter[count_idx] != 0:
            for j in range(784):
                avg[i][j] = sum_images_per_centroid[i][j] / counter[count_idx]
        count_idx += 1
    return avg


def not_a_random_init(centroids_not_random, x_train_np):
    centroids_not_random[0] = x_train_np[1]
    centroids_not_random[1] = x_train_np[3]
    centroids_not_random[2] = x_train_np[5]
    centroids_not_random[3] = x_train_np[7]
    centroids_not_random[4] = x_train_np[2]
    centroids_not_random[5] = x_train_np[0]
    centroids_not_random[6] = x_train_np[13]
    centroids_not_random[7] = x_train_np[15]
    centroids_not_random[8] = x_train_np[17]
    centroids_not_random[9] = x_train_np[4]
    return centroids_not_random


if __name__ == '__main__':
    run()
