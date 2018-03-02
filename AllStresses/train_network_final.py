# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 20:42:21 2018

@author: jgratsch
"""

# train the network for multi-class classficiation ############################

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.utils import resample
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
import matplotlib

# initialize the number of epochs to train for, initial learning rate,
# and batch size
dataset = 'images'
model_path = 'AllStressesModel_upsample_dropout.h5'
plot = 'Accuracy_plot_upsample_dropout'

# read in data ################################################################
# initialize the data and labels
print("[INFO] loading images...")
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
    if label == 'CornBorer':
        label = 0
    elif label == 'GreyLeafSpot':
        label = 1
    elif label == 'MagnesiumPotassiumDeficiency':
        label = 2
    elif label == 'NitrogenDeficiency':
        label = 3
    elif label == 'NorthernCornLeafBlight':
        label = 4
    elif label == 'PhosphorusDeficiency':
        label = 5
    elif label == 'Rust':
        label = 6
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# upsample minority class to resolve class imbalance problem ##################
cornBorer = np.array(np.where(labels == 0)).flatten()
GreyLeafSpot = np.array(np.where(labels == 1)).flatten()
MagnesiumPotassiumDeficiency = np.array(np.where(labels == 2)).flatten()
NitrogenDeficiency = np.array(np.where(labels == 3)).flatten()
NorthernCornLeafBlight = np.array(np.where(labels == 4)).flatten()
PhosphorusDeficiency = np.array(np.where(labels == 5)).flatten()
Rust = np.array(np.where(labels == 6)).flatten()

data0 = data[cornBorer]
data1 = data[GreyLeafSpot]
data2 = data[MagnesiumPotassiumDeficiency]
data3 = data[NitrogenDeficiency]
data4 = data[NorthernCornLeafBlight] # goal for upsample - 956
data5 = data[PhosphorusDeficiency]
data6 = data[Rust]

data0_resample = resample(data0,
                          replace = True,
                          n_samples = 956,
                          random_state = 123)
data1_resample = resample(data1,
                          replace = True,
                          n_samples = 956,
                          random_state = 123)
data2_resample = resample(data2,
                          replace = True,
                          n_samples = 956,
                          random_state = 123)
data3_resample = resample(data3,
                          replace = True,
                          n_samples = 956,
                          random_state = 123)
data5_resample = resample(data5,
                          replace = True,
                          n_samples = 956,
                          random_state = 123)
data6_resample = resample(data6,
                          replace = True,
                          n_samples = 956,
                          random_state = 123)

data_resampled = np.concatenate([data0_resample, data1_resample, data2_resample,
                                 data3_resample, data4, data5_resample, data6_resample], axis = 0)

label_0 = np.full((956,), 0)
label_1 = np.full((956,), 1)
label_2 = np.full((956,), 2)
label_3 = np.full((956,), 3)
label_4 = np.full((956,), 4)
label_5 = np.full((956,), 5)
label_6 = np.full((956,), 6)

labels_resampled = np.concatenate([label_0, label_1, label_2, label_3, label_4,
                                   label_5, label_6], axis = 0)

###############################################################################
# train the network
def train(model_path, plot):
    EPOCHS = 25
    INIT_LR = 1e-3
    BS = 32
    
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data_resampled,
    	labels_resampled, test_size=0.25, random_state=42)
         
    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=7)
    testY = to_categorical(testY, num_classes=7)
    
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    	horizontal_flip=True, fill_mode="nearest")
    
    # initialize the model
    print("[INFO] compiling model...")
    model = LeNet.build(width=28, height=28, depth=3, classes=7)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
    	metrics=["accuracy"])
    
    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    	epochs=EPOCHS, verbose=1)
     
    # save the model to disk
    print("[INFO] serializing network...")
    model.save(model_path)
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on All Stresses")
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
    
    
    


