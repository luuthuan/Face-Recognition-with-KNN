# Face Recognition using K-Nearest Neighbors (KNN)

## Overview

This project implements a basic face recognition system using the K-Nearest Neighbors (KNN) algorithm. The code reads training and test datasets of images, processes them, trains a KNN classifier, and makes predictions on the test data.

## Dependencies

- OpenCV (`cv2`)
- NumPy
- scikit-learn

## Code Description

### Functions

1. **read_data(path="datasets/singers/train/", size=(32,32))**
   - Reads images from the specified training directory.
   - Converts images to grayscale and resizes them to the given size.
   - Assigns a unique label to each folder of images.
   - Returns arrays of images and their corresponding labels.

2. **train_knn(X_vector, y)**
   - Trains a KNN classifier with the given image vectors and labels.
   - Uses 1-nearest neighbor for classification.
   - Returns the trained KNN model.

3. **read_test_data(path="datasets/singers/test/", size=(32,32))**
   - Reads and processes images from the specified test directory.
   - Converts images to grayscale and resizes them to the given size.
   - Returns an array of processed test images.

### Main Execution Flow

1. **Reading Data**
   - Reads and processes training data using `read_data()`.
   - Reads and processes test data using `read_test_data()`.

2. **Vectorization**
   - Converts the 3D image arrays (number of images, width, height) into 2D arrays (number of images, width \* height) for both training and test data.

3. **Training and Prediction**
   - Trains the KNN classifier using the vectorized training data.
   - Makes predictions on the vectorized test data.
   - Prints the predicted labels for the test images.

### Example Usage

To run the code, simply execute the script:

```bash
python face_recognition.py
