import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


import numpy as np 
import os
import sys

import matplotlib.pyplot as plt  
import pandas as pd 
import seaborn as sns
import cv2
import IPython
from six.moves import urlib

# We scale these values to a range of 0 to 1 before feeding to the neural network model.
#  For this, we divide the values by 255. It's important that the training set and the testing set are
#  preprocessed in the same way:
def preprocess_images(imgs): 
    sample_img = imgs if len(imgs.shape) == 2 else imgs[0]
    assert sample_img.shape in [(28, 28, 1), (28, 28)], sample_img.shape 
    return imgs / 255.0


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    model = keras.Sequential()
    # 32 convolution filters used each of size 3x3
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    # 64 convolution filters used each of size 3x3
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    # flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    # fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    # one more dropout
    model.add(Dropout(0.5))
    # output a softmax to squash the matrix into output probabilities
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=5)
    # test on test set
    print(test_images.shape)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)