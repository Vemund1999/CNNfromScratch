import numpy as np

import os 
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

from cnn import CNN
from convl import ConvL
from dl import DL
from nn import NN
from normalization import Normalization
from flatten import Flatten
from utils import *

import random


def main():
    # Paths for the datasets
    path_digit_training = r"/home/vemund/Downloads/Digit_training/"
    path_digit_testing = r"/home/vemund/Downloads/Digit_testing/"
        
    # Making the datasets
    x_train, y_train = get_dataset(path_digit_training)
    x_test, y_test = get_dataset(path_digit_testing)



    # model parameters
    epochs = 3
    learn_rate = 0.01

   # calculation of how large the input is, based off of how it will change throughout the convoluational layers.
    input_dim = (28-1)*(28-1)*3
    output_dim = 10

    # Making a model. Specifying it's configuration
    nn = CNN(
        ConvL(kernel_size=(3,3), n_kernels=3), # Convolutional layer
        Flatten(), # Flattening the data
        Normalization(), # Normelizing

        DL(n_neurons=input_dim, activation="sigmoid"), # Three dense neuron layers
        DL(n_neurons=20, activation="sigmoid"),
        DL(n_neurons=output_dim, activation="sigmoid")
    )
    # Training the model
    nn.train(x_train, y_train, epochs, learn_rate)

    accuracies = nn.final_accuracy_over_each_epoch()
    for i in range(len(accuracies)):
        print(f"Accuracy for epoch {i+1} : {round( accuracies[i]*100, 2)}½")

    # Plotting total amount of accuracies over all epochs
    plot_accuracies(nn.accuracies_over_time, "accuracies_over_time.png", "Accuracies over time", "Runs", "Accuracy")
    # Plotting accuracies with respect to epochs
    plot_accuracies(nn.accuracies_over_epochs, "accuracies_over_epochs.png", "Accuracy over epochs", "Runs", "Accuracy")

    # Printing the runtime of different layers
    print_runtimes( nn.get_runtimes() )
    # Plotting the kernels for the last image the model is trained on
    plot_kernels( nn.kernels_over_time )
    # Plotting what that image looks like after the kernels have been applied
    plot_images_from_kernels( nn.images_from_kernels_at_last_index , x_train[len(x_train) - 1] )





    # predicting single image
    rand_i = random.randrange( len(x_test) - 1 )
    rand_img = x_test[rand_i]
    rand_l = y_test[rand_i]
    cv2.imwrite("plots/image_to_predict.png", rand_img)
    pred = nn.predict(rand_img)
    print(f"Random image was predicted to be {pred}. It's actual value was {rand_l}")

    # testing the model
    acc = nn.test(x_test, y_test)
    print(f"Accuracy on test dataset is {round(acc*100, 2)}½")




main()

