
import numpy as np



# Layer for normelizing the values
class Normalization:

    # normelizing the values
   def forward(self, x):
       x = x / np.max(np.abs(x))
       return x
   
   # passing the data further back through the layers, without changing this layer
   def backward(self, delta):
       return delta




