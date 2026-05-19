# Handwritten Digit Recognition using CNN

## Project Overview

This project is a Deep Learning based Handwritten Digit Recognition system built using a Convolutional Neural Network (CNN) with TensorFlow and Keras.

The model is trained on the famous MNIST Dataset dataset to recognize handwritten digits from **0 to 9** with high accuracy.

The project demonstrates how CNNs can be used for image classification tasks in computer vision.

---

## Features

* Handwritten digit classification (0–9)
* Deep Learning model using CNN
* Image preprocessing and normalization
* Multiple convolution and pooling layers
* Dropout regularization to reduce overfitting
* Model saving using `.h5` format
* Beginner-friendly implementation

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

---

## Dataset

This project uses the MNIST Dataset dataset, which contains:

* 60,000 training images
* 10,000 testing images
* Grayscale handwritten digit images
* Image size: 28 × 28 pixels

---

## Model Architecture

The CNN model consists of:

1. Convolutional Layer (32 filters)
2. Max Pooling Layer
3. Convolutional Layer (64 filters)
4. Max Pooling Layer
5. Flatten Layer
6. Dense Layer (128 neurons)
7. Dropout Layer
8. Output Layer (10 classes with Softmax)

---

## Project Workflow

### 1. Import Required Libraries

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
```

### 2. Load Dataset

```python
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
```

### 3. Data Preprocessing

* Normalize pixel values
* Reshape images for CNN input

```python
X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
```

### 4. Build CNN Model

```python
cnn = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(128,activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(10,activation='softmax')
])
```

### 5. Compile and Train Model

```python
cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn.fit(X_train, y_train, epochs=10)
```

### 6. Evaluate Model

```python
cnn.evaluate(X_test, y_test)
```

### 7. Save Trained Model

```python
cnn.save("digit_model.h5")
```

---

## Model Performance

The model achieves high accuracy on the MNIST test dataset after training for 10 epochs.

Example metrics:

* Training Accuracy: ~99%
* Testing Accuracy: ~98%

(Accuracy may vary slightly during each run.)

---

## Folder Structure

```bash
DigitRecognition/
│
├── digit_model.h5
├── main.py
├── README.md
└── requirements.txt
```

---

## Installation

### Clone Repository

```bash
git clone <your-github-repository-link>
cd DigitRecognition
```

### Create Virtual Environment

```bash
python -m venv tfenv
```

### Activate Environment

#### Linux / Mac

```bash
source tfenv/bin/activate
```

#### Windows

```bash
tfenv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

```txt
tensorflow
numpy
matplotlib
```

---

## Future Improvements

* Build a Streamlit web application
* Upload custom handwritten images for prediction
* Improve model accuracy using data augmentation
* Deploy the project online
* Add real-time digit recognition

---

## Applications

* OCR systems
* Bank cheque digit recognition
* Postal code recognition
* Educational AI applications
* Document digitization

---

## Conclusion

This project showcases the power of Convolutional Neural Networks (CNNs) in image classification tasks. It is an excellent beginner-friendly Deep Learning project to understand computer vision concepts and neural network architecture.
