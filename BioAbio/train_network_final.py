# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 20:42:21 2018

@author: jgratsch
"""

# train the network ##########################################################
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import h5py
import tensorflow as tf
from sklearn.utils import resample
import matplotlib

# initialize variables
dataset = 'images'
model_path = 'AbioBioModel_upsample_dropout.h5'
plot = 'Accuracy_plot_upsample_dropout'

# initialize the data and labels
print("Running: loading images")
data = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "AbioticStress" else 0
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# upsample minority class to resolve class imbalance problem ##################
majority = np.reshape(np.where(labels == 0),(1720,))
minority = np.reshape(np.where(labels == 1),(446,))

abioticData = data[minority]
bioticData = data[majority]

abioticData_resample = resample(abioticData,
                                replace = True,
                                n_samples = 1720,
                                random_state = 123)

data_resampled = np.concatenate([abioticData_resample, bioticData], axis = 0)

label_1 = np.full((1720,), 1)
label_0 = np.full((1720,), 0)

labels_resampled = np.concatenate([label_1, label_0], axis = 0)

###############################################################################

# train network
#def train(model_path, plot):
# initialize the number of epochs to train for, initial learning rate,
# and batch size
def train(model_path, plot):    
    EPOCHS = 25
    INIT_LR = 1e-3
    BS = 32
     
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data_resampled,
    	labels_resampled, test_size=0.25, random_state=42)
     
    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)
    
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    	horizontal_flip=True, fill_mode="nearest")
    
    # initialize the model
    print("Running: compiling model.")
    model = LeNet.build(width=28, height=28, depth=3, classes=2)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
    	metrics=["accuracy"])
     
    # train the network
    print("Running: training network.")
    
    # fit the model
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    	epochs=EPOCHS, verbose=1)
     
    # save the model
    print("Running: serializing network.")
    model.save(model_path)
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Abiotic Stress/Biotic Stress")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot)
    
# calculate confusion matrix ##################################################
import pandas as pd
y_pred = model.predict(testX)
y_true = testY.copy()

y_pred = pd.DataFrame(y_pred)
y_true = pd.DataFrame(y_true)

y_pred['class'] = y_pred.idxmax(axis=1)
y_true['class'] = y_true.idxmax(axis=1)

confusion_matrix(y_true['class'],y_pred['class'])

# cross validation ############################################################
from sklearn.model_selection import StratifiedKFold
X = data_resampled
Y = labels_resampled
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)
cvscores = []
for train, test in kfold.split(X,Y):
    
    # convert the labels from integers to vectors
    trainY = to_categorical(Y[train], num_classes=7)
    testY = to_categorical(Y[test], num_classes=7)
    
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")
    print("[INFO] compiling model...")
    
    model2 = LeNet.build(width=28, height=28, depth=3, classes=7)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model2.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    
    # train the network
    print("[INFO] training network...")
    model2.fit_generator(aug.flow(X[train], trainY, batch_size=BS),
                            steps_per_epoch=len(X[train]) // BS,
                            epochs=EPOCHS, verbose=1)
    
    scores = model2.evaluate(X[test],testY, verbose = 1)
    print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))  