# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:32:19 2019

@author: Hrishikesh S
"""

# cd "Desktop/Third Year/Machine Learning/Project"

import os
import json
import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
import lpr_image_processing

def build_model():
    """
    build CNN model
    padded input shape is (40, 140)
    """
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='linear',
                     input_shape=(40, 140, 1),
                     padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),
                           padding='same'))
    model.add(Conv2D(64,
                     (3, 3),
                     activation='linear',
                     padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           padding='same'))
    model.add(Conv2D(128,
                     (3, 3),
                     activation='linear',
                     padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           padding='same'))
    model.add(Flatten())
    model.add(Dense(128,
                    activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(29,
                    activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def train_model(model,
                train_x,
                train_y,
                batch_size,
                epochs):
    """
    train the model
    model       - Sequential() added with all layers
    train_x     - input features for training
    train_x     - labels for training
    batch_size  - batch size
    epochs      - number of epochs the model should train for
    """
    history = model.fit(train_x,
                      train_y,
                      batch_size=batch_size,
                      epochs=epochs)
    return model, history

def train_and_save_model():
    """
    train the model
    """
    # listing all folders
    folder_list = os.listdir('images/segregated')
    # loading the images
    individual_images, labels = lpr_image_processing.final_extraction(folder_list)
    pad_x, pad_y = lpr_image_processing.determine_max_row_and_column_size(individual_images)
    # padding by resize
    x_train = lpr_image_processing.image_padding_by_resize(individual_images, pad_x, pad_y)
    # encode class values as integers
    # one hot encoding
    encoder = LabelEncoder()
    encoder.fit(labels)
    y_encoded = encoder.transform(labels)
    # convert integers into categorical values
    # by converting it into a bit array form (i.e. one hot encoded)
    y_train = np_utils.to_categorical(y_encoded)
    x_train = np.reshape(x_train, (-1, 40, 140, 1))
    lpr_model = build_model()
    lpr_model.summary()
    lpr_model, _ = train_model(lpr_model, x_train, y_train, 100, 10)
    # save model
    # serialize model to JSON
    model_json = lpr_model.to_json()
    with open("models/lpr.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    lpr_model.save_weights("models/lpr.h5")
    print("Saved model to disk")
