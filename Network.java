// Author: Jonathan Trahan
// Date: 10/27/2022

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.DoubleStream;
import java.util.Arrays;

public class Network {
	// instance variables
	double learning_rate;
	int batch_size;
	int epoch;
	int[] layers;
	double[][] weights1;
	double[][] weights2;
	double[] bias1;
	double[] bias2;
	double[][][] weights1_grad; // [mini batch size][1st dimension of weight1][2nd dimension of weight1]
	double[][][] weights2_grad; // [mini batch size][1st dimension of weight2][2nd dimension of weight2]
	double[][] bias1_grad; // [mini batch size][dimension of bias1]
	double[][] bias2_grad; // [mini batch size][dimension of bias1]
	int[] correct_digits;
	int[] total_digits;
	boolean trained;
	
	// Network class constructer
	public Network(double learning_rate, int batch_size, int epoch, int[] layers, double[][] weights1, double[][] weights2, double[] bias1, double[] bias2, double[][][] weights1_grad, double[][][] weights2_grad, double[][] bias1_grad, double[][] bias2_grad, int[] correct_digits, int[] total_digits, boolean trained) {
		this.learning_rate = learning_rate;
		this.batch_size = batch_size;
		this.epoch = epoch;
		this.layers = layers;
		this.weights1 = weights1;
		this.weights2 = weights2;
		this.bias1 = bias1;
		this.bias2 = bias2;
		this.weights1_grad = weights1_grad;
		this.weights2_grad = weights2_grad;
		this.bias1_grad = bias1_grad;
		this.bias2_grad = bias2_grad;
		this.correct_digits = correct_digits;
		this.total_digits = total_digits;
		this.trained = trained;
	}

	public static void main(String[] args) {
		// initialize constant values
		double learning_rate = 1;
		int batch_size = 10; 
		int epoch = 30; // number of times the network is trained
		int[] layers = {784, 30, 10}; // the number of layers and the number of nodes in the layers 
		
		// get random weights between input layer and hidden layer
		double[][] weights1 = new double[layers[1]][layers[0]];
		for (int i = 0; i < weights1.length; i++)
			for (int j = 0; j < weights1[i].length; j++)
				weights1[i][j] = Math.random()*2 - 1; // random double between -1 (inclusive) and 1 (exclusive)
		// get random weights between hidden layer and final layer
		double[][] weights2 = new double[layers[2]][layers[1]];
		for (int i = 0; i < weights2.length; i++)
			for (int j = 0; j < weights2[i].length; j++)
				weights2[i][j] = Math.random()*2 - 1; // random double between -1 (inclusive) and 1 (exclusive)
		
		// get random bias for hidden layer
		double[] bias1 = new double[layers[1]];
		for (int i = 0; i < bias1.length; i++)
			bias1[i] = Math.random()*2 - 1; // random double between -1 (inclusive) and 1 (exclusive)
		// get random bias for final layer
		double[] bias2 = new double[layers[2]];
		for (int i = 0; i < bias2.length; i++)
			bias2[i] = Math.random()*2 - 1; // random double between -1 (inclusive) and 1 (exclusive)
		
		// the first dimension for the gradient arrays is the batch_size and the other dimensions match the weights and biases arrays
		double[][][] weights1_grad = new double[batch_size][layers[1]][layers[0]];
		double[][][] weights2_grad = new double[batch_size][layers[2]][layers[1]];
		double[][] bias1_grad = new double[batch_size][layers[1]];
		double[][] bias2_grad = new double[batch_size][layers[2]];
		
		// these will always be an array of length 10
		// default value is 0
		int[] correct_digits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		int[] total_digits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		
		// value to specify if the network has been trained
		// set to false by default
		boolean trained = false;
		
		// create new network object with above default values
		Network n1 = new Network(learning_rate, batch_size, epoch, layers, weights1, weights2, bias1, bias2, weights1_grad, weights2_grad, bias1_grad, bias2_grad, correct_digits, total_digits, trained);
		
		// print out title (your name, your student number, the date, the assignment number, and a brief description of what the program does.)
		System.out.println("");
		System.out.println("Description: This program is a MNIST Handwritten Digit Recognizer Neural Network coded in Java.");
		System.out.println("");
		
		// start the loop
		loop(n1);
		
	}
	
	// the loop brings back the main menu after you finish the selected option from the menu
	// so that you don't have to restart the program
	public static void loop(Network net) {
		// create an instance of data
		int[][] data = null;
		
		// loop forever or until break is called
		while (true) {
			// display the main menu to get user input
			int choice = mainMenu();
			
			// load and process data based on user input
			// Train The Network
			if (choice == 1) {
				// load training data
				data = loadData(net, "train");
				
				// create index array used to randomize the data
				int[] indexArray = new int[data.length];
				for (int i = 0; i < data.length; i++)
					indexArray[i] = i;
				
				// format the data into a double array and scale the pixel values to 0-1
				double[][] new_data = formatInput(net, data);
				
				// train the network
				SGD(net, indexArray, new_data);
				
				// set network to trained
				net.trained = true;
			}
			// Load a Pre-Trained Network
			else if (choice == 2) {
				// load pre-trained weight sets and bias sets
				// load weights1 into net
				double[][] we1 = loadWeights(net, "weights1");
				net.weights1 = we1;
				
				// load weights2 into net
				double[][] we2 = loadWeights(net, "weights2");
				net.weights2 = we2;
				
				// load bias1 into net
				double[] bi1 = loadBias(net, "bias1");
				net.bias1 = bi1;
				
				// load bias2 into net
				double[] bi2 = loadBias(net, "bias2");
				net.bias2 = bi2;
				
				// set network to trained
				net.trained = true;
			}
			// Display TRAINING Data Accuracy
			else if (choice == 3 && net.trained == true) {
				// load training data
				data = loadData(net, "train");
				
				// format the data into double array and pixel values scaled to 0-1
				double[][] new_data = formatInput(net, data);
				
				// use displayAccuracy to count the total and correct digits
				displayAccuracy(net, new_data);
				
				// statistics is what prints out the accuracy
				statistics(net);
			}
			// if choice is equal to 3 but the network is not trained then pass by other options and show menu again
			else if (choice == 3 && net.trained == false) {
				System.out.println("only available after selecting [1] or [2] above");
			}
			//  Display TESTING Data Accuracy
			else if (choice == 4 && net.trained == true) {
				// load test data
				data = loadData(net, "test");
				
				// format the data into double array and pixel values scaled to 0-1
				double[][] new_data = formatInput(net, data);
				
				// use displayAccuracy to count the total and correct digits
				displayAccuracy(net, new_data);
				
				// statistics is what prints out the accuracy
				statistics(net);
			}
			// if choice is equal to 4 but the network is not trained then pass by other options and show menu again
			else if (choice == 4 && net.trained == false) {
				System.out.println("only available after selecting [1] or [2] above");
			}
			// Save Current Weight Set
			else if (choice == 5 && net.trained == true) {
				save(net);
			}
			// if choice is equal to 5 but the network is not trained then pass by other options and show menu again
			else if (choice == 5 && net.trained == false) {
				System.out.println("only available after selecting [1] or [2] above");
			}
			// Exit the loop
			else if (choice == 0) {
				break;
			}
			
		}
		
	}
	
	// main menu for the program where you choose what to do
	public static int mainMenu() {
		// print out the main menu
		System.out.println("");
		System.out.println("Main Menu");
		System.out.println("\t[1] Train The Network");
		System.out.println("\t[2] Load a Pre-Trained Network");
		System.out.println("\t[3] Display TRAINING Data Accuracy");
		System.out.println("\t[4] Display TESTING Data Accuracy");
		System.out.println("\t[5] Save Current Weight Set");
		System.out.println("\t[0] Exit");
		System.out.println("");
		
		// get input from user
		System.out.print("Select a number: ");
		String choice = System.console().readLine();
		System.out.println("");
		
		// parse input as an integer and if input is not an integer then set to -1
		int foo;
		try {
		   foo = Integer.parseInt(choice);
		}
		catch (NumberFormatException e) {
		   foo = -1;
		}
		
		// check if input is a valid option
		if (foo >= 0 && foo <= 6) {
			return foo;
		}
		else {
			// recursivly call menu until a correct option is selected
			System.out.println("Choose a correct option.");
			foo = mainMenu();
		}
		
		// return user input
		return foo;
	}
	
	// Stochastic Gradient Descent function
	public static void SGD(Network net, int[] index, double[][] data) {
		// train the network for the number of epochs
		for (int k = 0; k < net.epoch; k++) {
			System.out.println("##################");
			System.out.print("SGD epoch number: ");
			System.out.print(k);
			System.out.print("\n");
			System.out.println("##################");
			
			// reset the values of correct_digits and total_digits to 0
			int[] c_digits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
			int[] t_digits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
			net.correct_digits = c_digits;
			net.total_digits = t_digits;
			
			// Randomize the order of the items in the index array
			int[] rand_index = randArray(index);
			
			// Divide the Randomized data into mini-batches
			int[][] batch_index = batchConvert(rand_index, net.batch_size);
			
			// backpropagation
			// for loop through the rows of batch_index
			// rows are of length batch_size so each row is an array with the indexes of the inputs in each mini batch
			for (int i = 0; i < batch_index.length; i++){
				for (int j = 0; j < batch_index[i].length; j++) {
					backProp(net, data[batch_index[i][j]], j);
				}
				// update after each mini batch
				update(net);
			}
			// print out the statistics for the epoch
			statistics(net);
		}
	}
	
	// backpropagation function used in SGD
	public static void backProp(Network net, double[] input, int index_in_batch) {
		// separate input array into label and pixels (or input array)
		int label = ((int) input[0]);
		double[] one_hot = oneHot(label);
		double[] pixels = new double[784];
		for (int i = 0; i < pixels.length; i++) {
			pixels[i] = input[i+1];
		}
		
		// increment the total number for each digit
		net.total_digits[label]++;
		
		// forward pass
		// get the activation array (act1) for the hidden layer
		double[] act1 = new double[net.bias1.length];
		act1 = feedForward(net.weights1, net.bias1, pixels);
		
		// get the activation array (act2) for the final layer
		double[] act2 = new double[net.bias2.length];
		act2 = feedForward(net.weights2, net.bias2, act1);
		
		// get the largest value in the final layers activation array (also the one-hot vector that the network guessed)
		// the index of the largest value is the digit that the network guessed
		int largest = 0;
		for (int i = 0; i < act2.length; i++) {
			if (act2[i] > act2[largest])
				largest = i;
		}
		int guessed_digit = largest;
		
		// check if the guessed digit is the correct digit
		if (guessed_digit == label) {
			// if correct, increment the total number correct_digits for each digit
			net.correct_digits[label]++;
		}
		
		// backward pass 
		// get the bias gradient and weight gradient for the final layer
		// bias gradient
		double[] bi2_grad = new double[net.bias2.length];
		for (int i = 0; i < bi2_grad.length; i++) {
			bi2_grad[i] = (act2[i] - one_hot[i]) * act2[i] * (1 - act2[i]);
		}
		
		// weights gradient
		double[][] wei2_grad = new double[bi2_grad.length][act1.length];
		for (int i = 0; i < bi2_grad.length; i++) {
			for (int j = 0; j < act1.length; j++) {
				wei2_grad[i][j] = act1[j] * bi2_grad[i];
			}
		}
		
		// get the bias gradient and weight gradient for the hidden layer
		// bias gradient
		double[] bi1_grad = new double[net.bias1.length];
		for (int i = 0; i < bi1_grad.length; i++) {
			double sum = 0.0;
			for (int j = 0; j < bi2_grad.length; j++) {
				sum = sum + (net.weights2[j][i] * bi2_grad[j]);
			}
			bi1_grad[i] = sum * (act1[i] * (1 - act1[i]));
		}
		
		// weights gradient
		double[][] wei1_grad = new double[bi1_grad.length][pixels.length];
		for (int i = 0; i < bi1_grad.length; i++) { // 30
			for (int j = 0; j < pixels.length; j++) { // 784
				wei1_grad[i][j] = pixels[j] * bi1_grad[i];
			}
		}
		
		// store the bias and weight gradients in their matching class variables
		net.weights1_grad[index_in_batch] = wei1_grad;
		net.weights2_grad[index_in_batch] = wei2_grad;
		net.bias1_grad[index_in_batch] = bi1_grad;
		net.bias2_grad[index_in_batch] = bi2_grad;
		
	}
	
	// feedforward function used to calculate the activation vector for a layer 
	// given a layer's weights, biases, and the activations of the previous layer
	public static double[] feedForward(double[][] weight, double[] bias, double[] input) {
		// activation vector
		double[] activations = new double[bias.length];
		
		// calculate z to be used in the sigmoid formula
		double z = 0.0;
		for (int i = 0; i < weight.length; i++) {
			z = 0.0;
			for (int j = 0; j < weight[i].length; j++) {
				z = z + (weight[i][j]*input[j]);
			}
			z = z + bias[i];
			// store z for use below
			activations[i] = z;
		}
		
		// use z in sigmoid function (1/(1+e^-z))
		for (int i = 0; i < activations.length; i++) {
			activations[i] = 1/(1+Math.exp(-(activations[i]))); // e = 2.718281828459045 (value used by the Math library)
			//activations[i] = 1/(1+Math.pow(2.71828, -(activations[i]))); // e = 2.71828 (value used in the excel network)
		}
		
		// return the activation array
		return activations;
	}
	
	// function to update the weights and bias
	public static void update(Network net) {
		// temp var sum used to get the sum of the weight/bias gradients
		// sun_arr is the array to store the sum of each bias's gradients from each mini batch
		double[] sum_arr1 = new double[net.bias1_grad[0].length]; // net.bias1_grad[0].length == net.bias1
		double sum = 0.0;
		
		// get the sum of the bias gradients for each bias
		for (int i = 0; i < net.bias1_grad[0].length; i++) {
			sum = 0.0;
			for (int j = 0; j < net.bias1_grad.length; j++) {
				sum = sum + net.bias1_grad[j][i];
			}
			sum_arr1[i] = sum;
		}
		
		// update bias1
		for (int i = 0; i < net.bias1.length; i++) {
			// update bias at i
			net.bias1[i] = net.bias1[i] - (net.learning_rate/((double) net.batch_size)) * sum_arr1[i];
		}
		
		// sun_arr is the array to store the sum of each bias's gradients from each mini batch
		double[] sum_arr2 = new double[net.bias2_grad[0].length];
		
		// get the sum of the bias gradients for each bias
		for (int i = 0; i < net.bias2_grad[0].length; i++) { // # of biases in bias2 (final layer)
			sum = 0.0;
			for (int j = 0; j < net.bias2_grad.length; j++) { // batch_size
				sum = sum + net.bias2_grad[j][i];
			}
			sum_arr2[i] = sum;
		}
		
		// update bias2
		for (int i = 0; i < net.bias2.length; i++) {
			// update bias at i
			net.bias2[i] = net.bias2[i] - (net.learning_rate/((double) net.batch_size)) * sum_arr2[i];
		}
		
		
		// array used to store the sum of the gradients for weights1
		double[][] sum_arr3 = new double[net.weights1.length][net.weights1[0].length];
		
		// get the sum of the gradients for each weight
		sum = 0.0;		
		for (int i = 0; i < net.weights1_grad[0].length; i++) {
			for (int j = 0; j < net.weights1_grad[0][i].length; j++) {
				for (int k = 0; k < net.weights1_grad.length; k++) {
					sum = sum + net.weights1_grad[k][i][j];
				}
				sum_arr3[i][j] = sum;
				sum = 0.0;
			}
		}
		
		// update weights1
		for (int i = 0; i < sum_arr3.length; i++) {
			for (int j = 0; j < sum_arr3[i].length; j++) {
				// update weights1[i][j]
				net.weights1[i][j] = net.weights1[i][j] - (net.learning_rate/((double) net.batch_size)) * sum_arr3[i][j];
			}
		}
		
		// array used to store the sum of the gradients for weights2
		double[][] sum_arr4 = new double[net.weights2.length][net.weights2[0].length];
		
		// get the sum of the gradients for each weight
		sum = 0.0;
		for (int i = 0; i < net.weights2.length; i++) { // 2
			for (int j = 0; j < net.weights2[i].length; j++) { // 3
				for (int k = 0; k < net.weights2_grad.length; k++) { // 2 (batch_size)
					sum = sum + net.weights2_grad[k][i][j];
				}
				sum_arr4[i][j] = sum;
				sum = 0.0;
			}
		}
		
		// update weights2
		for (int i = 0; i < net.weights2.length; i++) {
			for (int j = 0; j < net.weights2[i].length; j++) {
				// update weights2[i][j]
				net.weights2[i][j] = net.weights2[i][j] - (net.learning_rate/((double) net.batch_size)) * sum_arr4[i][j];
			}
		}
		
	}
	
	// convert a double into the one-hot array of that number
	public static double[] oneHot(double label) {
		// create new one_hot array
		double[] one_hot = new double[10];
		
		// if the the digit is a 0 then replace it with a one-hot array of 0
		if (label == 0.0) {
			one_hot[0] = 1.0;
			one_hot[1] = 0.0;
			one_hot[2] = 0.0;
			one_hot[3] = 0.0;
			one_hot[4] = 0.0;
			one_hot[5] = 0.0;
			one_hot[6] = 0.0;
			one_hot[7] = 0.0;
			one_hot[8] = 0.0;
			one_hot[9] = 0.0;
		}
		// if the the digit is a 1 then replace it with a one-hot array of 1
		else if (label == 1.0) {
			one_hot[0] = 0.0;
			one_hot[1] = 1.0;
			one_hot[2] = 0.0;
			one_hot[3] = 0.0;
			one_hot[4] = 0.0;
			one_hot[5] = 0.0;
			one_hot[6] = 0.0;
			one_hot[7] = 0.0;
			one_hot[8] = 0.0;
			one_hot[9] = 0.0;
		}
		// if the the digit is a 2 then replace it with a one-hot array of 2
		else if (label == 2.0) {
			one_hot[0] = 0.0;
			one_hot[1] = 0.0;
			one_hot[2] = 1.0;
			one_hot[3] = 0.0;
			one_hot[4] = 0.0;
			one_hot[5] = 0.0;
			one_hot[6] = 0.0;
			one_hot[7] = 0.0;
			one_hot[8] = 0.0;
			one_hot[9] = 0.0;
		}
		// if the the digit is a 3 then replace it with a one-hot array of 3
		else if (label == 3.0) {
			one_hot[0] = 0.0;
			one_hot[1] = 0.0;
			one_hot[2] = 0.0;
			one_hot[3] = 1.0;
			one_hot[4] = 0.0;
			one_hot[5] = 0.0;
			one_hot[6] = 0.0;
			one_hot[7] = 0.0;
			one_hot[8] = 0.0;
			one_hot[9] = 0.0;
		}
		// if the the digit is a 4 then replace it with a one-hot array of 4
		else if (label == 4.0) {
			one_hot[0] = 0.0;
			one_hot[1] = 0.0;
			one_hot[2] = 0.0;
			one_hot[3] = 0.0;
			one_hot[4] = 1.0;
			one_hot[5] = 0.0;
			one_hot[6] = 0.0;
			one_hot[7] = 0.0;
			one_hot[8] = 0.0;
			one_hot[9] = 0.0;
		}
		// if the the digit is a 5 then replace it with a one-hot array of 5
		else if (label == 5.0) {
			one_hot[0] = 0.0;
			one_hot[1] = 0.0;
			one_hot[2] = 0.0;
			one_hot[3] = 0.0;
			one_hot[4] = 0.0;
			one_hot[5] = 1.0;
			one_hot[6] = 0.0;
			one_hot[7] = 0.0;
			one_hot[8] = 0.0;
			one_hot[9] = 0.0;
		}
		// if the the digit is a 6 then replace it with a one-hot array of 6
		else if (label == 6.0) {
			one_hot[0] = 0.0;
			one_hot[1] = 0.0;
			one_hot[2] = 0.0;
			one_hot[3] = 0.0;
			one_hot[4] = 0.0;
			one_hot[5] = 0.0;
			one_hot[6] = 1.0;
			one_hot[7] = 0.0;
			one_hot[8] = 0.0;
			one_hot[9] = 0.0;
		}
		// if the the digit is a 7 then replace it with a one-hot array of 7
		else if (label == 7.0) {
			one_hot[0] = 0.0;
			one_hot[1] = 0.0;
			one_hot[2] = 0.0;
			one_hot[3] = 0.0;
			one_hot[4] = 0.0;
			one_hot[5] = 0.0;
			one_hot[6] = 0.0;
			one_hot[7] = 1.0;
			one_hot[8] = 0.0;
			one_hot[9] = 0.0;
		}
		// if the the digit is a 8 then replace it with a one-hot array of 8
		else if (label == 8.0) {
			one_hot[0] = 0.0;
			one_hot[1] = 0.0;
			one_hot[2] = 0.0;
			one_hot[3] = 0.0;
			one_hot[4] = 0.0;
			one_hot[5] = 0.0;
			one_hot[6] = 0.0;
			one_hot[7] = 0.0;
			one_hot[8] = 1.0;
			one_hot[9] = 0.0;
		}
		// if the the digit is a 9 then replace it with a one-hot array of 9
		else if (label == 9.0) {
			one_hot[0] = 0.0;
			one_hot[1] = 0.0;
			one_hot[2] = 0.0;
			one_hot[3] = 0.0;
			one_hot[4] = 0.0;
			one_hot[5] = 0.0;
			one_hot[6] = 0.0;
			one_hot[7] = 0.0;
			one_hot[8] = 0.0;
			one_hot[9] = 1.0;
		}
		
		// return the one-hot array
		return one_hot;
	}
	
	// format the input to a double array and scale the pixels to between 0 and 1
	public static double[][] formatInput(Network net, int[][] input) {
		// create new 2d double array to store the formatted input
		double[][] new_input = new double[input.length][input[0].length];
		
		// interate through the input and convert all values to doubles
		// scale the pixels to 0-1
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[i].length; j++) {
				if (j == 0) {
					// convert the label to double
					new_input[i][j] = ((double) input[i][j]);
				}
				else if (j > 0) {
					// scale the pixel value from 0-255 to 0-1
					new_input[i][j] = ((double) input[i][j])/255;
				}
				
			}
		}
		
		// return the formatted input
		return new_input;
	}
	
	// convert randomized index array into a mini batch array
	public static int[][] batchConvert(int[] indexArr, int batch_size) {
		// create new 2d int array to store the mini batch indexes
		int[][] batchArr = new int[indexArr.length / batch_size][batch_size];
		
		// interate through the batchArr and indexArr and copy values to the batchArr from the indexArr
		int k = 0;
		for (int i = 0; i < batchArr.length; i++) {
			for (int j = 0; j < batchArr[i].length; j++, k++) {
				batchArr[i][j] = indexArr[k];
			}
		}
		
		// return the batchArr
		return batchArr;
	}
	
	// randomize array function
	// takes an int array and returns an int array with the jumbled up elements of the input array
	public static int[] randArray(int[] ar) {
		// create Random variable rand
		Random rand = ThreadLocalRandom.current();
		
		// randomize input array by swaping values at random index
		for (int i = 0; i < ar.length; i++) {
			int index = rand.nextInt(i + 1);
			// swap
			int a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
		}
		
		return ar;
	}
	
	// loads the data set based on the name given
	public static int[][] loadData(Network net, String set) {
		// create file variables
		File f = null;
		String[][] strdata = null;
		int[][] intdata = null;
		
		// load training data
		if (set == "train") {
			f = new File("mnist_train.csv");
			strdata = new String[60000][785];
			intdata = new int[60000][785];
		}
		// load testing data
		else if (set == "test") {
			f = new File("mnist_test.csv");
			strdata = new String[10000][785];
			intdata = new int[10000][785];
		}
		
		// create scanner variable which is used to read through the csv files
		int i = 0;
		try {
			Scanner sc = new Scanner(f);
			// break if there is no next line
			while (sc.hasNextLine()) {
				String[] line = sc.nextLine().split(",");
				strdata[i] = line;
				i++;
			}
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
		
		// parse string array as int array
		for (i = 0; i < strdata.length; i++) {
			for (int j = 0; j < strdata[i].length; j++) {
				intdata[i][j] = Integer.parseInt(strdata[i][j]);
			}
		}
		
		// count the total number of each digit
		// reset the values of total_digits
		int[] t_digits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		net.total_digits = t_digits;
		for (i = 0; i < intdata.length; i++) {
			// count 0's
			if (intdata[i][0] == 0)
				net.total_digits[0]++;
			// count 1's
			else if (intdata[i][0] == 1)
				net.total_digits[1]++;
			// count 2's
			else if (intdata[i][0] == 2)
				net.total_digits[2]++;
			// count 3's
			else if (intdata[i][0] == 3)
				net.total_digits[3]++;
			// count 4's
			else if (intdata[i][0] == 4)
				net.total_digits[4]++;
			// count 5's
			else if (intdata[i][0] == 5)
				net.total_digits[5]++;
			// count 6's
			else if (intdata[i][0] == 6)
				net.total_digits[6]++;
			// count 7's
			else if (intdata[i][0] == 7)
				net.total_digits[7]++;
			// count 8's
			else if (intdata[i][0] == 8)
				net.total_digits[8]++;
			// count 9's
			else if (intdata[i][0] == 9)
				net.total_digits[9]++;
		}
		
		return intdata;
	}
	
	// loads the weight sets based on the name given
	public static double[][] loadWeights(Network net, String set) {
		// create instance of f, strdata, doubledata
		File f = null;
		String[][] strdata = null;
		double[][] doubledata = null;
		
		// if set is weights1 then load the file weights1.csv
		// weights1 has dimensions [30][784]
		if (set == "weights1") {
			f = new File("weights1.csv");
			strdata = new String[30][784];
			doubledata = new double[30][784];
		}
		// if set is weights2 then load the file weights2.csv
		// weights2 has dimensions [10][30]
		else if (set == "weights2") {
			f = new File("weights2.csv");
			strdata = new String[10][30];
			doubledata = new double[10][30];
		}
		
		// create index variable i
		int i = 0;
		try {
			// create scanner variable to read through file
			Scanner sc = new Scanner(f);
			// loops as long as there is a next line in the file
			while (sc.hasNextLine()) { 
				// reads the next line and splits it into a string array using "," as the delimiter
				String[] line = sc.nextLine().split(",");
				strdata[i] = line;
				i++;
			}
			
		} 
		catch (Exception e) { // catch the exception e
			e.printStackTrace();
		}
		 // convert the string array into a double array
		for (i = 0; i < strdata.length; i++)
			for (int j = 0; j < strdata[i].length; j++)
				doubledata[i][j] = Double.parseDouble(strdata[i][j]);
		
		// return the double array
		return doubledata;
	}
	
	// loads the bias sets based on the name given
	public static double[] loadBias(Network net, String set) {
		// create instance of f, strdata, doubledata
		File f = null;
		String[] strdata = null;
		double[] doubledata = null;
		
		// if set is bias1 then load the file bias1.csv
		// bias1 has dimensions [30]
		if (set == "bias1") {
			f = new File("bias1.csv");
			strdata = new String[30];
			doubledata = new double[30];
		}
		// if set is bias2 then load the file bias2.csv
		// bias2 has dimensions [10]
		else if (set == "bias2") {
			f = new File("bias2.csv");
			strdata = new String[10];
			doubledata = new double[10];
		}
		
		// create index variable i
		int i = 0;
		try {
			// create scanner variable to read through file
			Scanner sc = new Scanner(f);
			// loops as long as there is a next line in the file
			while (sc.hasNextLine()) { 
				// reads the next line and splits it into a string array using "," as the delimiter
				String[] line = sc.nextLine().split(",");
				strdata = line;
				i++;
			}
			
		} 
		catch (Exception e) { // catch the exception e
			e.printStackTrace();
		}
		 // convert the string array into a double array
		for (i = 0; i < strdata.length; i++)
			doubledata[i] = Double.parseDouble(strdata[i]);
		
		// return the double array
		return doubledata;
	}

	// function used to print out the statistics after each epoch
	public static void statistics(Network net) {
		System.out.println();
		// correct/total for digit 0
		System.out.print("0 = ");
		System.out.print(net.correct_digits[0]);
		System.out.print("/");
		System.out.print(net.total_digits[0]);
		System.out.print("\t");

		// correct/total for digit 1
		System.out.print("1 = ");
		System.out.print(net.correct_digits[1]);
		System.out.print("/");
		System.out.print(net.total_digits[1]);
		System.out.print("\t");
		
		// correct/total for digit 2
		System.out.print("2 = ");
		System.out.print(net.correct_digits[2]);
		System.out.print("/");
		System.out.print(net.total_digits[2]);
		System.out.print("\t");
		
		// correct/total for digit 3
		System.out.print("3 = ");
		System.out.print(net.correct_digits[3]);
		System.out.print("/");
		System.out.print(net.total_digits[3]);
		System.out.print("\t");
		
		// correct/total for digit 4
		System.out.print("4 = ");
		System.out.print(net.correct_digits[4]);
		System.out.print("/");
		System.out.print(net.total_digits[4]);
		System.out.print("\t");
		
		// correct/total for digit 5
		System.out.print("5 = ");
		System.out.print(net.correct_digits[5]);
		System.out.print("/");
		System.out.print(net.total_digits[5]);
		System.out.print("\t");
		
		// start a new line
		System.out.print("\n");
		
		// correct/total for digit 6
		System.out.print("6 = ");
		System.out.print(net.correct_digits[6]);
		System.out.print("/");
		System.out.print(net.total_digits[6]);
		System.out.print("\t");
		
		// correct/total for digit 7
		System.out.print("7 = ");
		System.out.print(net.correct_digits[7]);
		System.out.print("/");
		System.out.print(net.total_digits[7]);
		System.out.print("\t");
		
		// correct/total for digit 8
		System.out.print("8 = ");
		System.out.print(net.correct_digits[8]);
		System.out.print("/");
		System.out.print(net.total_digits[8]);
		System.out.print("\t");
		
		// correct/total for digit 9
		System.out.print("9 = ");
		System.out.print(net.correct_digits[9]);
		System.out.print("/");
		System.out.print(net.total_digits[9]);
		System.out.print("\t");
		
		// get the sum of correct_digits
		int sum1 = 0;
		for (int i = 0; i < net.correct_digits.length; i++)
			sum1 = sum1 + net.correct_digits[i];
		
		// get the sum of total_digits
		int sum2 = 0;
		for (int i = 0; i < net.total_digits.length; i++)
			sum2 = sum2 + net.total_digits[i];
		
		// calculate the accuracy
		double accuracy = ((double)sum1)/((double)sum2);
		
		// print out the accuracy (correct/total)
		System.out.print(" Accuracy = ");
		System.out.print(sum1);
		System.out.print("/");
		System.out.print(sum2);
		System.out.print(" = ");
		System.out.print(accuracy*100.0);
		System.out.print("%");
		System.out.println();
		System.out.println();
	}
	
	public static void displayAccuracy(Network net, double[][] data) {
		// reset the values of correct_digits and total_digits
		int[] c_digits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		int[] t_digits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		
		net.correct_digits = c_digits;
		net.total_digits = t_digits;
		
		for (int k = 0; k < data.length; k++) {
			double[] input = data[k];
			
			// separate input array into label and pixels (or input array)
			int label = ((int) input[0]);
			double[] pixels = new double[784];
			for (int i = 0; i < pixels.length; i++) {
				pixels[i] = input[i+1];
			}
			
			// increment total digits at label
			net.total_digits[label]++;
			
			// get the activation vector for the hidden layer
			double[] act1 = new double[net.layers[1]];
			act1 = feedForward(net.weights1, net.bias1, pixels); // 30x1 change input back to pixels
			
			// get the activation vector for the final layer
			double[] act2 = new double[net.layers[2]];
			act2 = feedForward(net.weights2, net.bias2, act1); // 10x1
			
			// get the largest value in the final layers activation vector (also the one-hot vector that the network guessed)
			// the index of the largest value is the digit that the network guessed
			int largest = 0;
			for (int i = 0; i < act2.length; i++) {
				if (act2[i] > act2[largest])
					largest = i;
			}
			int guessed_digit = largest;
			
			// check if the guessed digit is the correct digit
			if (guessed_digit == label) {
				// increment correct_digits at guessed_digit
				net.correct_digits[guessed_digit]++;
			}
		}
		
	}
	
	public static void save(Network net) {
		// convert weights1 to String
		String[][] weights1_str = new String[net.weights1.length][net.weights1[0].length];
		for (int i = 0; i < net.weights1.length; i++)
			for (int j = 0; j < net.weights1[i].length; j++)
				weights1_str[i][j] = String.valueOf(net.weights1[i][j]); // convert double to String
		
		// convert weights2 to String
		String[][] weights2_str = new String[net.weights2.length][net.weights2[0].length];
		for (int i = 0; i < net.weights2.length; i++)
			for (int j = 0; j < net.weights2[i].length; j++)
				weights2_str[i][j] = String.valueOf(net.weights2[i][j]); // convert double to String
		
		// convert bias1 to String
		String[] bias1_str = new String[net.bias1.length];
		for (int i = 0; i < net.bias1.length; i++)
			bias1_str[i] = String.valueOf(net.bias1[i]); // convert double to String
		
		// convert bias2 to String
		String[] bias2_str = new String[net.bias2.length];
		for (int i = 0; i < net.bias2.length; i++)
			bias2_str[i] = String.valueOf(net.bias2[i]); // convert double to String
		
		
		// save weights1
		// create csv file
		try {
			File csvFile1 = new File("weights1.csv"); // create csv file named weights1.csv
			// check if csv file was created
			if (csvFile1.createNewFile()) {
				System.out.println("File created: " + csvFile1.getName());
			} 
			else { // if file was not created then the file already exists
				System.out.println("weights1.csv already exists.");
			}
		} 
		catch (IOException e) { // catch the IOException error
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		
		// write to csv file
		try {
			FileWriter myWriter1 = new FileWriter("weights1.csv"); // create myWriter1 to write to file named weights1.csv
			// interate through weights1_str and write it to weights1.csv
			for (int i = 0; i < weights1_str.length; i++) {
				for (int j = 0; j < weights1_str[i].length; j++) {
					// if weights1_str[i][j] is the last value in the row then you don't need a comma
					if (weights1_str[i][j] == weights1_str[i][net.weights1[0].length - 1]) {
						myWriter1.write(weights1_str[i][j]);
					}
					// if weights1_str[i][j] is the last value in weights1_str then you can break out of the for loop
					else if (weights1_str[i][j] == weights1_str[net.weights1.length - 1][net.weights1[0].length - 1]) {
						myWriter1.write(weights1_str[i][j]);
						break;
					}
					// if weights1_str[i][j] is a normal value in weights1_str then write a comma after it
					else {
						myWriter1.write(weights1_str[i][j]);
						myWriter1.write(",");
					}
				}
				// write a new line after every row
				myWriter1.write("\n");
			}
			// close myWriter1
			myWriter1.close();
			System.out.println("Successfully wrote to weights1.csv");
		}
		catch (IOException e) { // catch the IOException error
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		
		
		// save weights2
		try {
			File csvFile2 = new File("weights2.csv"); // create csv file named weights2.csv
			// check if csv file was created
			if (csvFile2.createNewFile()) {
				System.out.println("File created: " + csvFile2.getName());
			} 
			else { // if file was not created then the file already exists
				System.out.println("weights2.csv already exists.");
			}
		} 
		catch (IOException e) { // catch the IOException error
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		
		// write to csv file
		try {
			FileWriter myWriter2 = new FileWriter("weights2.csv"); // create myWriter2 to write to file named weights2.csv
			// interate through weights2_str and write it to weights2.csv
			for (int i = 0; i < weights2_str.length; i++) {
				for (int j = 0; j < weights2_str[i].length; j++) {
					// if weights2_str[i][j] is the last value in the row then you don't need a comma
					if (weights2_str[i][j] == weights2_str[i][net.weights2[0].length - 1]) {
						myWriter2.write(weights2_str[i][j]);
					}
					// if weights2_str[i][j] is the last value in weights2_str then you can break out of the for loop
					else if (weights2_str[i][j] == weights2_str[net.weights2.length - 1][net.weights2[0].length - 1]) {
						myWriter2.write(weights2_str[i][j]);
						break;
					}
					// if weights2_str[i][j] is a normal value in weights2_str then write a comma after it
					else {
						myWriter2.write(weights2_str[i][j]);
						myWriter2.write(",");
					}
				}
				// write a new line after every row
				myWriter2.write("\n");
			}
			// close myWriter2
			myWriter2.close();
			System.out.println("Successfully wrote to weights2.csv");
		}
		catch (IOException e) { // catch the IOException error
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		
		
		// save bias1
		try {
			File csvFile3 = new File("bias1.csv"); // create csv file named bias1.csv
			// check if csv file was created
			if (csvFile3.createNewFile()) {
				System.out.println("File created: " + csvFile3.getName());
			} 
			else { // if file was not created then the file already exists
				System.out.println("bias1.csv already exists.");
			}
		} 
		catch (IOException e) { // catch the IOException error
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		
		// write to csv file
		try {
			FileWriter myWriter3 = new FileWriter("bias1.csv"); // create myWriter3 to write to file named bias1.csv
			// interate through bias1_str and write it to bias1.csv
			for (int i = 0; i < bias1_str.length; i++) {
				// if bias1_str[i] is the last value in the row then you don't need a comma
				if (bias1_str[i] == bias1_str[net.bias1.length - 1]) {
					myWriter3.write(bias1_str[i]);
				}
				// if bias1_str[i] is a normal value in bias1_str then write a comma after it
				else {
					myWriter3.write(bias1_str[i]);
					myWriter3.write(",");
				}
			}
			// close myWriter3
			myWriter3.close();
			System.out.println("Successfully wrote to bias1.csv");
		}
		catch (IOException e) { // catch the IOException error
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		
		
		// save bias2
		try {
			File csvFile4 = new File("bias2.csv"); // create csv file named bias2.csv
			// check if csv file was created
			if (csvFile4.createNewFile()) {
				System.out.println("File created: " + csvFile4.getName());
			} 
			else { // if file was not created then the file already exists
				System.out.println("bias2.csv already exists.");
			}
		} 
		catch (IOException e) { // catch the IOException error
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		
		// write to csv file
		try {
			FileWriter myWriter4 = new FileWriter("bias2.csv"); // create myWriter4 to write to file named bias2.csv
			// interate through bias2_str and write it to bias2.csv
			for (int i = 0; i < bias2_str.length; i++) {
				// if bias2_str[i] is the last value in the row then you don't need a comma
				if (bias2_str[i] == bias2_str[net.bias2.length - 1]) {
					myWriter4.write(bias2_str[i]);
				}
				// if bias2_str[i] is a normal value in bias2_str then write a comma after it
				else {
					myWriter4.write(bias2_str[i]);
					myWriter4.write(",");
				}
			}
			// close myWriter4
			myWriter4.close();
			System.out.println("Successfully wrote to bias2.csv");
		}
		catch (IOException e) { // catch the IOException error
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
		
	}
	
	// print out weights1, weights2, bias1, and bias2
	public static void printWeightsBias(Network net) {
		// print out weights1
		System.out.println();
		System.out.println("weights1 = ");
		for (int i = 0; i < net.weights1.length; i++) {
			System.out.print("{");
			for (int j = 0; j < net.weights1[i].length; j++) {
				System.out.print(net.weights1[i][j]);
				System.out.print(", ");
			}
			System.out.print("}");
			System.out.print("\n");
		}
		System.out.println();
		
		// print out weights2
		System.out.println();
		System.out.println("weights2 = ");
		for (int i = 0; i < net.weights2.length; i++) {
			System.out.print("{");
			for (int j = 0; j < net.weights2[i].length; j++) {
				System.out.print(net.weights2[i][j]);
				System.out.print(", ");
			}
			System.out.print("}");
			System.out.print("\n");
		}
		System.out.println();
		
		// print out bias1
		System.out.println();
		System.out.println("bias1 = ");
		System.out.print("{");
		for (int i = 0; i < net.bias1.length; i++) {
			System.out.print(net.bias1[i]);
			System.out.print(", ");
		}
		System.out.print("}");
		System.out.print("\n");
		System.out.println();
		
		// print out bias2
		System.out.println();
		System.out.println("bias2 = ");
		System.out.print("{");
		for (int i = 0; i < net.bias2.length; i++) {
			System.out.print(net.bias2[i]);
			System.out.print(", ");
		}
		System.out.print("}");
		System.out.print("\n");
		System.out.println();
	}
	
}