import cv2
import os
from sklearn.neighbors import KNeighborsClassifier

import numpy as np


# Define a read datasets functions
def read_data(path="datasets/singers/train/", size=(32, 32)):
    # Define images and label array
    images, labels = [], []
    # Define a temp label
    temp_label = 0
    tmp = {}
    for folder_name in os.listdir(path):
        temp_label += 1
        tmp[str(temp_label)] = folder_name
        detail_path = f"{path}{folder_name}"  # detail_path = datasets/singers/train/folder_name

        for file in os.listdir(detail_path):
            # Start reading for each image
            img = cv2.imread(f"{detail_path}/{file}", cv2.IMREAD_GRAYSCALE)
            # Start image scale
            if img is not None:
                img = cv2.resize(img, size)
                # Start add img to images
                images.append(img)
                # Start add temp_label to labels
                labels.append(temp_label)

    print(tmp)
    return np.array((images)), np.array(labels)


# Define a training model function with params are X_vector and y is labels
def train_knn(X_vector, y):
    knn = KNeighborsClassifier(n_neighbors=1)  # k = 1
    knn.fit(X_vector, y)
    return knn


# Define a reading data function for folder test
def read_test_data(path="datasets/singers/test/", size=(32, 32)):
    images = []
    for file in os.listdir(path):
        f_path = f"{path}{file}"
        print(f_path)

        im = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            im = np.resize(im, size)
            images.append(im)
    return np.array(images)


if __name__ == '__main__':
    # Start reading data of train and test folder
    X, y = read_data()
    X_test = read_test_data()
    # print(f"Labels: {y}")
    # print(f"Images: {X}")
    # print(X.shape) # (15, 32, 32) is 15 elements with 32x32
    # print(f"Images test: {X_test.shape}")
    # KNN Algorithms - K-nearest-neighbor (Thuật toán láng giềng gần với k phải là số lẻ)

    # Vectorization of train matrix (15 elements with 32x32) to (15 rows and 32x32 cols)
    nums_samples, width, height = X.shape
    X_vector = X.reshape(nums_samples, width * height)
    # print(X_vector.shape) # (15, 1024)

    # Vectorization of test matrix (4 elements with 32x32) to (4 rows and 32x32 cols)
    nums_samples, width, height = X_test.shape
    X_test_vector = X_test.reshape(nums_samples, width * height)
    # print(X_test_vector.shape) # (4, 1024)

    # Start train model
    knn = train_knn(X_vector, y)

    # Predict with data test
    y_predict = knn.predict(X_test_vector)
    print(y_predict)
