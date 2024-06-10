
from utils import *

import numpy as np

# The convolutional layer
class ConvL:
   def __init__(self, kernel_size, n_kernels):
       # initilazing variables for the conv layer
       self.kernel_height = kernel_size[0]
       self.kernel_width = kernel_size[1]
       self.n_kernels = n_kernels
       self.learning_rate = None # will be initilazed when initilizing the cnn.

                    # bounds for the initial kernel values # amount of kernels     # shape of the kernel                    
       self.kernels = np.random.uniform(-0.5, 0.5, size=(self.n_kernels, self.kernel_height,self.kernel_width)) # initilizing the kernels
       self.cache = None # for saving the original image before the kernels were applied


   def forward(self, data):
       self.cache = data # saving the image
       
       feature_maps = []
       for i in range(len(self.kernels)): # for each kernel
           x_height = data.shape[0]
           x_width = data.shape[1]
           feature_map = np.zeros((x_height - 1, x_width - 1)) # calculating the size of the feature map
           for j in range(x_height - self.kernel_height + 1): 
               for k in range(x_width - self.kernel_width + 1): 
                   region = data[j:j + self.kernel_height, k:k + self.kernel_width] # getting a region og the image
                   feature_map[j,k] = np.sum( region * self.kernels[i] ) # applying the kernel
           feature_maps.append(feature_map)

       return np.array(feature_maps)




   def backward(self, delta):      
       delta_kernels = np.zeros(self.kernels.shape) # initilizing the size of the output delta
       x_height, x_width = self.cache.shape # getting the shape of the original image from the forward pass

       for i in range(len(self.kernels)):  # for each kernel
           for j in range(x_height - self.kernel_height + 1):  
               for k in range(x_width - self.kernel_width + 1): 
                   region = self.cache[j:j + self.kernel_height, k:k + self.kernel_width] # get a region og the original image
                   delta_kernels[i] += delta[i, j, k] * region  # calulcate the delta
       self.kernels -= self.learning_rate * delta_kernels # update the kernels
       return delta_kernels





