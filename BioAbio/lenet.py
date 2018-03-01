# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras import backend as K
import tensorflow as tf

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        
        # if using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20,(5,5), padding ="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # dropout method to minimize overfitting
        model.add(Dropout(0.25))
        
        #second set of CONV => RELU => POOL layers
        model.add(Conv2D(50,(5,5), padding ="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # drop out method to minimize overfitting
        model.add(Dropout(0.25))
        
        # first (and only) set of FC => RELU Layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        # return network architecture
        return model
    