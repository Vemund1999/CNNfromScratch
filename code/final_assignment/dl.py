
from utils import *

import numpy as np


# a dense layer that can be in the nn neural network
# holds the neuron values for the layer, as well as the bias
class DL:
    def __init__(self, n_neurons, activation=None):
        self.n_neurons = n_neurons
        self.neurons = np.zeros((n_neurons, 1))
        self.activation = activation
        self.bias = None

    # for applying the activation function to the neurons
    def apply_activation(self):
        if self.activation == "sigmoid":
            self.neurons = 1 / (1 + np.exp(-self.neurons))

    # for getting the derivative of the activation function applied to the neurons
    def get_derivative_of_activation(self):
        if self.activation == "sigmoid":
            return self.neurons * (1 - self.neurons)
    
    # initilizing the datastructure for the bias. Is initilized during the initilization of the nn neural network
    def initilize_bias(self, n_neurons_front_layer):
        self.bias = np.zeros((n_neurons_front_layer, 1))
        
    


