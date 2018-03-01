# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 20:42:21 2018

@author: jgratsch
"""

# train
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
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
model_path = 'AllStressesModel2.h5'
plot = 'Accuracy_plot'
def train(dataset, model_path, plot):
    
    EPOCHS = 25
    INIT_LR = 1e-3
    BS = 32
     
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
     
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
    	labels, test_size=0.25, random_state=42)
     
    y_train = trainY
    
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
    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = dict(enumerate(class_weights))
    
    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    	epochs=EPOCHS, verbose=1, class_weight = class_weights)
     
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