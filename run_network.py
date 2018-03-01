# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:27:30 2018

@author: jgratsch
"""

# import the test_network function
from test_network import test_network

# specify image and model path
image = 'test_images/rust3.jpg'
model_path = 'AllStressesModel_final.h5'

# run the trained network
test_network(image, model_path)
