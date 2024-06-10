
from utils import *

import numpy as np


# Layer for flattening the data, so the neural network can handle it
class Flatten:
    def __init__(self):
        self.original_shapes = None # saving the shape of the originl data

    # flattening the data
    def forward(self, x):
        self.original_shapes = x.shape
        return x.flatten().reshape(-1, 1)

        
    # Reconstructing the data to it's original shape, so it can be passed from the neural network, to the convolutional layers
    def backward(self, delta):
        delta = delta.reshape(self.original_shapes)
        return delta
    


