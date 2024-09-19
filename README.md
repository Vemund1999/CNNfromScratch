
# About project
This project is a framework for a CNN, that is coded from scratch.
The CNN has convolutional layers to process images, and a neural network to classify the images.
There is code for training a CNN model, which means it has forward- and back propagation
for each layer.

In this document I can showcase some aspects of how a CNN works.
I can for example look at how images change throughout the layers, how the kernels and
neurons change throughout the training, etc


# Dataset
I have trained the model on the MNIST dataset.
The dataset consists of handwritten digits. The digits range from 0-9.
The model is trained to try to classify which digit is being displayed on the images.

There are two fields in the main.py file where the paths for the images can be specified.
The path specified should be at the folder contining the subfolders for the digits (subfolders for ex digit 1, 2 etc...)
For example if the path is some_folder/some_other_folder/Digit_testing/0
Then the specified path for the variable in main.py should be some_folder/some_other_folder/Digit_testing/
Example:
![bilde](https://github.com/user-attachments/assets/62588e2e-94a0-4222-a707-c083a1793a9f)



# Analysis
## Framework
I’ve coded a framework for making a CNN. A CNN can be configured when initializing it.
Here’s an example of a possible configuration:
![bilde](https://github.com/user-attachments/assets/a743e404-5b32-43a6-bf96-16ad6adce737)
The results of the CNN will somewhat vary, depending on:
- the configuration of the CNN.
- values initialized at random (for example weight values initialized to be -0.5 – 0.5).
For the results, I’ll use the configuration above

## Running time
Larger kernels, more neurons, and more layer will increase the running time, as there are more computations that
needs to be done.
I’m using the above configuration
![bilde](https://github.com/user-attachments/assets/781f27f1-f0ea-4496-b94e-fcc23d54180e)

This image shows the average runtime for forward- and back propagation in each layer.
The first number in the list is for forward propagation, and the last number is for backward propagation.
We can see that the convolutional layer has the largest running time.
The 2nd dense layer has the 2nd largest runtime.
The first layer has no running time for the forward propagation, because the only thing being done is to input the
already-exiting values from the convolutionl layers into the neurons.
But the running times for the other dense layers show the amount of time taken to update the
weights between the dense layers.
The times for the dense layers show in large part matrix multiplication between layers


## Accuracy
![bilde](https://github.com/user-attachments/assets/deaedfed-a06b-4f3e-8ab4-7426b46bb2c8)
![bilde](https://github.com/user-attachments/assets/c383b727-6d5f-4c41-91d5-73814b8b4e38)

The image shows the accuracy for each epoch.
The first image shows the accuracy over all runs.
The 2nd image shows the accuracy for each epoch. The first image’s ending accuracy is dragged down as the
inaccuracies of the other epochs will drag the overall accuracy score down.
In the 2nd image this effect doesn't occur.
The accuracies starts low. But as the model adjusts it’s weights during training, the model’s accuracy increases.
The accuracy eventually stagnates as the there is not much more optimization of the weights to be done.

## Prediction
There are also methods to predict a single image, and to test the model on testing data.
If the model predicts a single image from the testing dataset, it manages to label the image correctly
![bilde](https://github.com/user-attachments/assets/f91c5d36-efb7-435e-9349-59bee4687f5b)


## Kernels
![bilde](https://github.com/user-attachments/assets/0da16d19-4a29-44bc-8d3b-54c5d1b35dec)
![bilde](https://github.com/user-attachments/assets/4055109c-66c0-4895-8ebf-c02c80cf3cf1)
![bilde](https://github.com/user-attachments/assets/0d6f69f9-763a-4c6a-89b1-04c291fa9d61)
Here we can see the kernels in the convolutional layer. “Epoch 2” is really the last epoch 3, but the index is used
in the title (and indexes start at 0).
The kernels are applied over the images to draw out features of the images, so that the neural network can better
classify them.
We can see how different kernels captures different features of the images



The CNN has a good performance. It’s able train in order to classify the hand written digits.
It’s interesting to see how the weights and values change as the model is being trained, which leads to the
model’s accuracy increasing.

# Possible extensions
The CNN API can be extended to have more functionalities.
It can for example have more plots and graphs to showcase more aspects of the model.
It can be extended to have different types of layers, such as a pooling layer.
Currently it has these possible layers: Dense layer, flattening layer, normalization layer, convolutional layer.
It can be extended to have more possible activation functions, such as relu. Currently it only has the sigmoid
activation function.






