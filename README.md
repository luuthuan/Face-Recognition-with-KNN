# Face Recognition with KNN

This project demonstrates a basic implementation of face recognition using the K-Nearest Neighbors (KNN) algorithm. The implementation includes reading image datasets, preprocessing images, applying PCA for dimensionality reduction, training a KNN model, and evaluating its performance.

## Prerequisites

Ensure you have the following libraries installed:

- `opencv-python`
- `scikit-learn`
- `numpy`

You can install these libraries using pip:

```bash
pip install opencv-python scikit-learn numpy
```
## Project Structure

. <br>
├── datasets <br>
│   ├── singers <br>
│       &emsp;├── train <br>
│       &emsp;├── test <br>
├── face_recognition.py <br>
└── README.md

- `datasets/singers/train/`: Contains training images organized in subfolders for each class (singer).<br>
- `datasets/singers/test/`: Contains test images for prediction.

## Usage

1. **Read and preprocess data**: The `read_data` function reads the training images, converts them to grayscale, resizes them, and labels them. Similarly, the `read_test_data` function reads and preprocesses the test images.<br>
2. **Train the KNN model**: The `train_knn` function trains the KNN model using GridSearchCV to find the best value for `n_neighbors`.<br>
3. **Dimensionality reduction with PCA**: The script applies PCA to reduce the dimensionality of the image vectors.<br>
4. **Predict and evaluate**: The trained KNN model is used to predict the classes of the test images, and the performance of the model is evaluated using cross-validation.

## Running the Script
To run the script, use the following command:
```bash
python face_recognition_v2.py
```

## Example Code
```bash
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
import numpy as np
from collections import Counter


# Define a read datasets function
def read_data(path="datasets/singers/train/", size=(32, 32)):
    images, labels = [], []
    temp_label = 0
    tmp = {}
    for folder_name in os.listdir(path):
        temp_label += 1
        tmp[str(temp_label)] = folder_name
        detail_path = f"{path}{folder_name}"
        for file in os.listdir(detail_path):
            img = cv2.imread(f"{detail_path}/{file}", cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, size)
                images.append(img)
                labels.append(temp_label)
    print(tmp)
    return np.array(images), np.array(labels)


# Define a training model function with params are X_vector and y is labels
def train_knn(X_vector, y):
    # Define the parameter grid
    param_grid = {'n_neighbors': [1, 3, 5, 7]}
    # Perform GridSearchCV to find the best k
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=2)  # Điều chỉnh số lần chia nhỏ
    grid_search.fit(X_vector, y)
    # Use the best found parameter for k
    best_k = grid_search.best_params_['n_neighbors']
    knn = KNeighborsClassifier(n_neighbors=best_k)
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
            im = cv2.resize(im, size)
            images.append(im)
    return np.array(images)


if __name__ == '__main__':
    # Start reading data of train and test folder
    X, y = read_data()
    X_test = read_test_data()

    # Check number of samples in each class
    print(Counter(y))

    # Vectorization of train matrix
    nums_samples, width, height = X.shape
    X_vector = X.reshape(nums_samples, width * height)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(10, nums_samples, width * height))
    X_vector_pca = pca.fit_transform(X_vector)

    nums_samples, width, height = X_test.shape
    X_test_vector = X_test.reshape(nums_samples, width * height)
    X_test_vector_pca = pca.transform(X_test_vector)

    # Start train model
    knn = train_knn(X_vector_pca, y)

    # Predict with data test
    y_predict = knn.predict(X_test_vector_pca)
    print(y_predict)

    # Evaluate model using cross-validation
    scores = cross_val_score(knn, X_vector_pca, y, cv=2)  # Điều chỉnh số lần chia nhỏ
    print(f"Cross-validation accuracy: {scores.mean()}")
```

## Explanation
- `read_data`: Reads and preprocesses training images from the specified path.
- `train_knn`: Trains a KNN model using GridSearchCV to determine the best `n_neighbors` value.
- `read_test_data`: Reads and preprocesses test images.
- PCA is applied to reduce dimensionality before training the KNN model.
- The script prints out the predicted labels for the test images and evaluates the model using cross-validation.

## Notes
- Ensure the dataset directory structure matches the expected format.
- Adjust the image size `(32, 32)` in the `read_data` and `read_test_data` functions if necessary.

```bash
This `README.md` provides a comprehensive overview of your project, including prerequisites, project structure, usage instructions, and a detailed explanation of the code.
```