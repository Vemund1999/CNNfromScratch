

from utils import *
import numpy as np
import time


# The neural network part of the CNN. 
# Contains the dense layers in the cnn, as well as the biases for those dense layers, and the weights between the dense layers
class NN():
    def __init__(self, *layers):
        self._layers = layers  # neural layers
        self.weights = self.initilize_weights() # initilizing the weights between the neural layers
        self.initilize_biases() # initilizingt the bias neuron

        # For taking note of the runtimes of the layers
        self.forward_run_times = [ [] for i in range( len(self.layers) ) ]
        self.backward_run_times = [ [] for i in range( len(self.layers) ) ]


        # Initilizing each bias to be of the shape of dense layer in front
    def initilize_biases(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].initilize_bias( self.layers[i + 1].n_neurons )

        # Initilizing the weights as matrices. Columns equal to the n neurons behind, and rows equal to the n neuron in front.
    def initilize_weights(self):
        weights = []
        for i in range(1, len(self.layers)):
            n_col = self.layers[i - 1].n_neurons
            n_row = self.layers[i].n_neurons
            weight = np.random.uniform(-0.5, 0.5, (n_row, n_col)) # initilizing each weight to be of a random value between -0.5 and 0.5
            weights.append(weight)
        return np.array(weights, dtype=object)



    @property
    def layers(self):
        return list(self._layers)
        
    # for predicting a single image. forward(x) will return the predicted label of x
    def predict(self, x):
        return self.forward(x)
    
    # forward propagating the data in the neural network
    def forward(self, x):
        # Setting the first dense layer to have it's neuron values be the input (passed from the flattening layer after the convolution)
        self.layers[0].neurons = x
        for i in range(1, len(self.layers)): # passing the data through each dense layer
            start_time = time.time() 

            self.layers[i].neurons = self.layers[i - 1].bias + self.weights[i - 1] @ self.layers[i - 1].neurons # setting the neuron values for the layer
            self.layers[i].apply_activation() # applying the activation function

            end_time = time.time()
            self.forward_run_times[i].append(end_time - start_time) # recording the time it took to go through the layer

        # returning the predicted label at the last layer
        return np.argmax(self.layers[ len(self.layers) - 1].neurons)



    def backward(self, error, learn_rate):
        start_time = time.time()

        # different indexes
        last_w_i = len(self.weights) - 1
        second_last_l_i = len(self.layers) - 2
        last_b_i = len(self.layers) - 2

        self.weights[ last_w_i ] += -learn_rate * error @ np.transpose(self.layers[ second_last_l_i ].neurons) # updating the weights
        self.layers[ last_b_i ].bias += learn_rate * error  # updating the bias
        delta = error # setting the error for the other layers

        end_time = time.time()
        self.backward_run_times[ len(self.backward_run_times) - 1 ].append(end_time - start_time) # recording the time it too to do the calculations


        for i in reversed(range( len(self.layers) - 1)): # going through the layers backwards
            start_time = time.time()

            delta = np.transpose(self.weights[i]) @ delta * self.layers[i].get_derivative_of_activation() # setting the delta
            if (i != 0):
                self.weights[i - 1] += -learn_rate * delta @ np.transpose(self.layers[i - 1].neurons) # updating the weights
                self.layers[i - 1].bias += -learn_rate * delta # updating the bias

            end_time = time.time()
            self.backward_run_times[i].append(end_time - start_time) # recording the time it took to go through do the calculations

        # returning the error at the last neuron layer, so it can be used for backpropagation in the convolutional layers
        return delta 









