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


def build_model():
    model = keras.Sequential([
        Desne(20, activation = tf.nn.relu, input_shape=[len(train_features[0])]),
        Dense(1)
        ])
        model.compile(optimizer=tf.train.AdamOptimizer(), loss='mse', metrics=['mae', 'mse'])
        return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(seflf, epoch, logs):
        if epoch % 100 == 0: 
            print(' ')
        print ('.', end=' ')

    
def plot_history(hist):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [Thousand Dollars$^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
    plt.legend()
    plt.ylim([0,50])

if __name__ == '__main__':
    (train_features, train_labels), (test_features, test_labels) = keras.datasets.boston_house.load_data()
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)
    trin_features = (train_features - train_mean) / train_std

    model = build_model()
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(trian_features, train_labels, epoch=1000, verbose=0,validation_split = 0.1,
                                     callbacks=[early_stop, PrintDot()])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    rmse_final = np.sqrt(float(hist['val_mean_squared_error'].tail(1)))
    print()
    print('Final Root Mean Square Error on validation set: {}'.format(round(rmse_final, 3)))
    plot_history(hist)
    # test on test_set
    test_features_norm = (test_features - train_mean) / train_std
    mse, _, _ = model.evaluate(test_features_norm, test_labels)
    rmse = np.sqrt(mse)
    print('Root Mean Square Error on test set: {}'.format(round(rmse, 3)))
