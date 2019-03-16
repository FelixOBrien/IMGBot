import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import load_model
from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

TRAINING_DIRECTORY = "training_data/"
BATCH_SIZE = 128
NUM_CLASSES = os.
NUM_EPOCHS = 1500
WEIGHT_DECAY = 5e-4

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def alexnet():
    model = Sequential()
    model.add(Conv2D(96, (11,11), activation='relu', padding='same', input_shape=(64,64,3), kernel_regularizer=l2(WEIGHT_DECAY), name='conv1'))
    model.add(MaxPooling2D((3,3), name='pool1'))
    model.add(Conv2D(256, (5,5), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY), name='conv2'))
    model.add(MaxPooling2D((3,3), name='pool2'))

    model.add(Conv2D(384, (3,3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY), name='conv3'))
    model.add(Conv2D(384, (3,3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY), name='conv4'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(WEIGHT_DECAY), name='conv5'))
    model.add(MaxPooling2D((3,3), name='pool3'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), name='fc2'))
    model.add(Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(WEIGHT_DECAY), name='output'))