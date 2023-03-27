# MNIST-Handwritten-Digit-Recognizing-Neural-Network
MNIST Handwritten Digit Recognizing Neural Network written in Java.

I constructed a Java program that recognizes the MNIST digit set. The data set, in CSV
format, is available at https://pjreddie.com/projects/mnist-in-csv/. The format of each
line of data is: “label, pix-1-1, pix-1-2, pix1-3, … , pix-28-28” where label is a digit 0-9 and pix-X-Y is a
greyscale value from 0-255.

MNIST files need to be in the same directory as the program file.

## User-selectable operations:
**[1] Train the network**

In training mode, your program should iterate through the 60,000 item MNIST training data set. I used a learning rate of 1, a mini-batch size of 10, and 30 epochs; randomly initializing my weights and biases to the -1 to 1 range, and scaling my pixel inputs to the range of 0 – 1. After each training epoch, the program prints out statistics showing: (1) for each of the ten digits [0-9], the number of correctly classified inputs over the total number of occurrences of that digit; (2) the overall accuracy considering all ten digits. In other words, after each epoch, the output should look something like this:
  0 = 4907/4932 1 = 5666/5678 2 = 4921/4968 3 = 5034/5101 4 = 4839/4859 5 = 4472/4506
  6 = 4935/4951 7 = 5140/5175 8 = 4801/4842 9 = 4931/4988 Accuracy = 49646/50000 = 99.292%

**[2] Load a pre-trained network**

The program is able to load a weight set (previously generated) from a file.

**[3] Display network accuracy on TRAINING data {only available after selecting [1] or [2] above}**

Iterate over the 60,000 item MNIST training data set exactly once, using the current weight set, and output the statistics shown in [1] above.

**[4] Display network accuracy on TESTING data {only available after selecting [1] or [2] above}**

Iterate over the 10,000 item MNIST testing data set exactly once, using the current weight set, and output the statistics shown in [1] above.

**[5] Save the network state to file {only available after selecting [1] or [2] above}**

The program is able to save the current weight set to a file.

**[0] Exit**

This will exit the program.
