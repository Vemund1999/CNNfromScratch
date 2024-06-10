from utils import *

import time
import numpy as np

from dl import DL
from convl import ConvL
from flatten import Flatten
from nn import NN
from normalization import Normalization

# The CNN
# holds the code for the whole model datastructure, and the code for training the model
class CNN:
    def __init__(self, *layers):
        self._layers = layers
        # initilizing the nn, dividing the layers into non-neural-network layers, and neural network layers
        # after applying this function, the self.layers will no longer contain the dense layers, only the self.nn will.
        self.nn, self._layers = self.initilize_nn() 
        self.all_layers = self.layers + self.nn.layers


        # Different datastructures for making graphs and such of the model
        self.accuracies_over_time = [] 
        self.accuracies_over_epochs = [] 
        self.kernels_over_time = [] 
        self.images_over_time = []

        self.final_accuracy_over_each_epoch = []

        self.conv_run_times_forward = [[] for i in range(len(self.layers))] 
        self.conv_run_times_backward = [[] for i in range(len(self.layers))] 

        self.is_training_on_last_index = False
        self.images_from_kernels_at_last_index = []

    @property
    def layers(self):
        return list(self._layers)
    

    # takes the runtimes for each layer, and returns the average runtime for the layer, both for forward and backward propagation
    def get_runtimes(self):
        r_for = self.conv_run_times_forward + self.nn.forward_run_times
        r_back = self.conv_run_times_backward + self.nn.backward_run_times

        r_for = average_runtimes(r_for)
        r_back = average_runtimes(r_back)

        times = {}
        for layer, forw, backw in zip(self.all_layers, r_for, r_back):
            if isinstance(layer, DL):
                times[f"DenseLayer({layer.n_neurons})"] = [forw, backw]
            elif isinstance(layer, Flatten):
                times["Flatten"] = [forw, backw]
            elif isinstance(layer, Normalization):
                times["Normalization"] = [forw, backw]
            elif isinstance(layer, ConvL):
                times[f"ConvL(({layer.kernel_height},{layer.kernel_width}), {layer.n_kernels})"] = [forw, backw]
        
        return times


    # Setting the lerning rate for the convolutional layers
    def initilize_learning_rate_C(self, learning_rate):
        for i in range(len(self.layers)):
            if (isinstance(self.layers[i], ConvL)):
                self.layers[i].learning_rate = learning_rate

    # For predicting the single image. Forward returns the predicted label of x (an image)
    def predict(self, x):
        prediction = self.forward(x)
        return prediction

    # for getting the accuracy on the test dataset
    def test(self, x_data, y_data):
        correct_predictions = 0
        for x, y in zip(x_data, y_data):
            prediction = self.forward(x) # getting the predicted label
            if (prediction == y):
                correct_predictions+=1 # recording how many labels have been correctly predicted
        return correct_predictions / (len(x_data) - 1) # returning the accuracy



    
    # for initilizing the nn neural network part of the CNN
    def initilize_nn(self):
        nn_layers = []
        non_nn_layers = []
        # dividing the layers between the NN holding the dense layers, and the CNN holding the non-dense-layers
        for i in range(len(self.layers)):
            if (isinstance(self.layers[i], DL)):
                nn_layers.append(self.layers[i])
            else:
                non_nn_layers.append(self.layers[i])
        return NN(*nn_layers), non_nn_layers
    


    # for training the model
    def train(self, x_data, y_data, epochs, learn_rate):
        last_index = len(x_data) - 1 # this image will be used to eventually get plots of the kernels and such

        start_time = time.time()

        self.initilize_learning_rate_C(learn_rate)

        correct_predictions = 0
        i = 0
        j = 0
        total_correct = 0
        for epoch in range(epochs):
            for x, y in zip(x_data, y_data):
                self.is_training_on_last_index = (j == last_index) # checking if the image being processed is the last image

                y = encode(y) # onehot encoding the label

                prediction = self.forward(x) # forward passing the data, and getting the precited label back

                # noting the mount of correct predictions
                i+=1
                j+=1
                is_correct = prediction == np.argmax(y) 
                if (is_correct):
                    correct_predictions+=1
                    total_correct+=1
                self.accuracies_over_time.append( total_correct / i )
                self.accuracies_over_epochs.append( correct_predictions / j )


                output = self.nn.layers[ len(self.nn.layers) - 1 ].neurons # getting the output values on the last layer in the neural network
                error = self.nn.layers[ len(self.nn.layers) - 1 ].neurons - y # calculating the loss between the output and the label

                self.backward(output, error, learn_rate) # backpropagating the error

            self.final_accuracy_over_each_epoch.append(correct_predictions / x_data.shape[0])

            correct_predictions = 0
            j = 0

            end_time = time.time()
            self.running_time = end_time - start_time # recording the running time
            


                

    def forward(self, x):
        # passing through processing concolutional layers
        for i in range(len(self.layers)):
            start = time.time()
            # processing the image in some layer
            x = self.layers[i].forward(x)
            # getting the images and the kernels from the convolutional layer for the last image in the dataset
            if (self.is_training_on_last_index and isinstance(self.layers[i], ConvL)):
                self.images_from_kernels_at_last_index.append(x)
                self.kernels_over_time.append(self.layers[i].kernels)

            end = time.time()
            self.conv_run_times_forward[i].append(end - start) # saving the running time for the forward pass
        
        # passing the processed image from the convolutional part of the CNN, into the neural network part of the CNN
        prediction = self.nn.forward(x)
        return prediction # returning the precition made in the nn neural network

    def backward(self, output, error, learn_rate):
        # backpropagating through the neural network
        delta = self.nn.backward(error, learn_rate)

        # backpropagating through the convolutional layers
        for i in reversed( range( len(self.layers) ) ):
            start = time.time()

            delta = self.layers[i].backward(delta) # backpropagating, the getting back an error.

            end = time.time()
            self.conv_run_times_backward[i].append(end - start) # saving the running time for the layer






