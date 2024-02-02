# Flood Prediction Model Using PyTorch

This repository contains a machine learning project aimed at predicting flood events based on various input features. The model is built using PyTorch, a popular deep learning library that provides tools for building complex neural network architectures. This document outlines the steps taken to preprocess the data, construct the neural network model, train the model, and evaluate its performance on a test dataset.

## Dataset

The dataset used in this project contains several features that are relevant to predicting flood events. These features include environmental and geographical data points, among others. The target variable is a binary indicator of whether a flood event occurred.

## Data Preprocessing

1. **Loading the Data**: The dataset is loaded into a pandas DataFrame from a CSV file.
2. **Feature Selection**: Unnecessary columns such as 'state' and 'city' are dropped, focusing on numerical features for the prediction task.
3. **Normalization**: Input features are standardized to have a mean of 0 and a standard deviation of 1. This step improves the convergence speed of the training process and the overall performance of the model.
4. **Tensor Conversion**: Data is converted from numpy arrays to PyTorch tensors, which are suitable for model training in PyTorch.
5. **Dataset Splitting**: The dataset is split into training and testing sets to evaluate the model's performance on unseen data.

## Model Architecture

The neural network, named `FloodPredictor`, consists of the following layers:

- **Input Layer**: Accepts the number of features from the dataset.
- **Hidden Layer 1**: A fully connected layer with 1800 neurons, followed by a ReLU activation function.
- **Hidden Layer 2**: Another fully connected layer with 1164 neurons, also followed by a ReLU activation function.
- **Output Layer**: A single neuron with a sigmoid activation function to output a probability indicating the likelihood of a flood event.

## Training Process

The model is trained using the Adam optimizer and the Binary Cross-Entropy loss function, suitable for binary classification tasks. Training involves iterating over batches of the training dataset, computing the loss, and updating the model's weights to minimize the loss.

## Testing and Evaluation

After training, the model's performance is evaluated on the test set to determine its accuracy in predicting flood events. The accuracy metric is calculated by comparing the model's predictions to the actual labels in the test dataset.


This project provides a foundational approach to predicting flood events using neural networks and can be extended or modified to suit different datasets or prediction tasks.