# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 16:21:38 2018

@author: jgratsch
"""

# import the test_network function
from test_network import test_network

# specify image and model path
image = 'test_images/magdef.jpg'
model_path = 'AbioBioModel_final.h5'

# run the trained network
test_network(image, model_path)

