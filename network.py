import keras

from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import os


TRAINING_DIRECTORY = "training_data/"
BATCH_SIZE = 128
NUM_CLASSES = len(os.listdir(TRAINING_DIRECTORY))
NUM_EPOCHS = 120
WEIGHT_DECAY = 5e-4

#Images for the network
IMAGES = []
#Labels for the network
LABELS = []
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
imagePaths = getListOfFiles(TRAINING_DIRECTORY)
for imagePath in imagePaths:

    img= image.load_img(imagePath, color_mode ='rgb', target_size = (64, 64, 3)) 
    img = image.img_to_array(img)
    IMAGES.append(img)

    label = imagePath.split(os.path.sep)[-2]
    
    label = label.split("/")[-1]
    LABELS.append(label)
IMAGES = np.array(IMAGES, dtype="float") / 255.0
LABELS = np.array(LABELS)

(X_train, X_test, y_train, y_test) = train_test_split(IMAGES, LABELS, test_size=0.25, random_state=42)
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
def convertLabels(label):
    return encoder.classes_[label]
if __name__ == "__main__":
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
        return model

    model = alexnet()
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-2, momentum=0.9), metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir='alexnet')

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.10, callbacks=[tensorboard])

    score = model.evaluate(np.array(X_test), np.array(y_test))
    print('\nTest accuracy: {0:.4f}'.format(score[1]))
    model.save('model.h5')