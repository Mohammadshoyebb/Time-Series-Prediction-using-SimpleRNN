# Time Series Prediction using SimpleRNN

This repository demonstrates the implementation of a Simple Recurrent Neural Network (SimpleRNN) for time series prediction using TensorFlow and Keras. Below is a detailed description of the implementation:

## Overview

### 1. Importing Libraries

The necessary libraries are imported, including Pandas for data manipulation, NumPy for numerical operations, Matplotlib for visualization, and Keras for building and training the neural network model.

### 2. Generating Synthetic Time Series Data

Synthetic time series data is generated using a sine wave with added random noise. The data is stored in a Pandas DataFrame (`df`) and visualized using Matplotlib.

### 3. Data Preprocessing

#### Function: `convertToMatrix`

A utility function `convertToMatrix` is defined to convert the time series data into input-output pairs suitable for supervised learning. It creates sequences of input (`X`) and corresponding output (`Y`) by sliding a window (`step`) over the time series data.

### 4. Train-Test Split and Reshaping

The generated data is split into training and testing sets (`train` and `test`). The input-output pairs are created using the `convertToMatrix` function, and the input data (`trainX` and `testX`) is reshaped to be compatible with the SimpleRNN model (`(samples, timesteps, features)`).

### 5. Model Definition

#### Sequential Model: `SimpleRNN`

A Sequential model is defined using Keras, consisting of:
- A SimpleRNN layer with 32 units, `relu` activation function, and input shape `(1, step)`.
- A Dense layer with 8 units and `relu` activation.
- A Dense output layer with 1 unit for predicting the next value in the time series.

The model is compiled with Mean Squared Error (MSE) loss and RMSprop optimizer.

### 6. Model Training

The SimpleRNN model is trained using the training data (`trainX`, `trainY`) for 100 epochs with a batch size of 16. Training progress is displayed (`verbose=2`).

### 7. Prediction and Evaluation

#### Prediction and Evaluation

The trained model (`model`) is used to make predictions on both the training (`trainX`) and testing (`testX`) datasets. Predictions are concatenated (`predicted`) for visualization.

#### Evaluation

The model's performance is evaluated using the training data (`trainX`, `trainY`) and the Mean Squared Error (MSE) is computed and printed.

### 8. Visualization

Matplotlib is used to visualize:
- The original time series data (`df`) overlaid with the predicted values (`predicted`).

## Conclusion

This README.md provides a detailed overview of implementing a SimpleRNN for time series prediction using synthetic data. It covers data generation, preprocessing, model definition, training, evaluation, and visualization. This implementation serves as a foundational example for understanding SimpleRNNs in time series forecasting tasks.
