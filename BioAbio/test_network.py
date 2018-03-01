# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 20:42:21 2018

@author: jgratsch
"""

# test the network on new images
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2

def test_network(image, model_path):
    # load the image
    image = cv2.imread(image)
    orig = image.copy()
     
    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    # load the trained convolutional neural network
    print("RUNNING: loading network.")
    model = load_model(model_path)
     
    # classify the input image
    (BioticStress, AbioticStress) = model.predict(image)[0]
    
    # build the label
    label = "Abiotic Stress" if AbioticStress > BioticStress else "Biotic Stress"
    proba = AbioticStress if AbioticStress > BioticStress else BioticStress
    label = "{}: {:.2f}%".format(label, proba * 100)
     
    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
    	0.7, (0, 255, 0), 2)
     
    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
