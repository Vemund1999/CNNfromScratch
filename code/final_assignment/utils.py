import matplotlib.pyplot as plt
import cv2
import numpy as np
import os 




# printing the runtimes
def print_runtimes(runtimes):
    for runtime in runtimes:
        print(f"{runtime}: {runtimes[runtime]}")

# making a plot for each kernel form the cnn, which was applied to the last image the cnn was trained on
def plot_kernels(kernels):
    for epoch in range(len(kernels)):
        for k in range(len( kernels[epoch] )):
            plot_matrix( kernels[epoch][k] , f"Kernel {k} in epoch {epoch}" , f"kernel_{k}_epoch_{epoch}.png" )

# for plotting a matrix, which the kernels are represented by
def plot_matrix(matrix, title, filename):
   plt.imshow(matrix, cmap="viridis", interpolation="none")
   for i in range(matrix.shape[0]):
       for j in range(matrix.shape[1]):
           plt.text(i, j, f"{ round( matrix[i][j] , 3 ) }", ha="center", va="center", color="black")

   plt.title(title)
   plt.savefig(f"plots/{filename}")
   plt.close()


# saving the images the kernels above were applied to
def plot_images_from_kernels(images_from_kernels, image):
    # saving specific image
    cv2.imwrite(f"plots/image.png", image)
    for i in range(len(images_from_kernels)): # for each epoch
        for j in range(len(images_from_kernels[i])): # for each group of kernels
            cv2.imwrite(f"plots/image_from_epoch_{i+1}_kernel_{j}.png", images_from_kernels[i][j])

    


# for getting a dataset
def get_dataset(path):
    images = []
    labels = []
    # for each subfolder (ex 0, 5, 2) holding the images of a label
    for subfolder in os.listdir(path):
        # for each file in the subfolder
        for file in os.listdir(f"{path}/{subfolder}"):            
            image_path = os.path.join(f"{path}/{subfolder}", file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (28, 28))
            
            images.append(image) # add the image to the dataset
            labels.append(int(subfolder)) # add the label for the image

    images = np.array(images)
    labels = np.array(labels)
    # shuffeling the images in random order, so the images aren't in order
    indices = np.random.permutation(len(images))
    shuffle_images = images[indices]
    shuffle_labels = labels[indices]

    return shuffle_images, shuffle_labels

# for onehot encoding the label
def encode(value):
    y = [0] * 10
    y[value] = 1
    y = np.array( y ).reshape(-1, 1)
    return y






# for plotting yhe accuracies
def plot_accuracies(accuracies, filename, title, xlabel, ylabel):
   for i in range(len(accuracies)):
       accuracies[i] = accuracies[i]*100

   plt.figure(figsize=(10, 6))
   plt.plot(range(len(accuracies)), accuracies, marker='o', linestyle='-', color='b')
   plt.title(title)
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)

   plt.ylim(0, 100)
   plt.grid(True)
   plt.savefig('plots/' + filename)
   plt.close()



# getting the average runtime for a layer
def average_runtimes(runtimes):
   avg_runtimes = []
   for i in range(len(runtimes)):
       try:
           avg_run = sum(runtimes[i]) / len(runtimes[i])
       except:
           avg_run = 0
       avg_runtimes.append(avg_run)

   return avg_runtimes

