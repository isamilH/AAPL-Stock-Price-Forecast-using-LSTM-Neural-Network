# AAPL-Stock-Price-Forecast-using-LSTM-Neural-Network
## Overview
This project implements an LSTM (Long Short-Term Memory) neural network to forecast Apple Inc. (AAPL) stock prices using historical data from Yahoo Finance. The model is trained to predict future stock prices based on past trends.
## Methodology
 1. Data Collection
        Historical AAPL stock data is downloaded from Yahoo Finance using the yfinance library.
        Data range: January 1, 2023, to September 17, 2024.
 2. Data Preprocessing
        Select the 'Close' price and reset the index.
        Resample the data to ensure a regular daily frequency and forward-fill missing values.
        Normalize the data using MinMaxScaler from scikit-learn.
 3. Sequence Creation
        Create sequences of 60 time steps to be used as input features for the LSTM model.
 4. Data Splitting
        Split the dataset into training (80%) and testing (20%) sets.
 5. Model Building
        Use TensorFlow's Keras API to build a Sequential LSTM model.
        The model architecture includes two LSTM layers and one Dense layer.
 6. Model Training
        Compile the model using the 'adam' optimizer and 'mean_squared_error' loss function.
        Train the model for 20 epochs with a batch size of 32.
 7. Prediction and Evaluation
        Make predictions on both training and testing datasets.
        Inverse transform the predictions to the original scale.
        Calculate Root Mean Squared Error (RMSE) for model evaluation.
 8. Visualization
        Plot the actual vs. predicted stock prices to visualize the model's performance.
 9. Future Forecasting
        Use the last 60 days of data to predict the stock price for the next day.

## Results
   Train RMSE: 4.52
   Test RMSE: 7.03

The model demonstrates the ability to capture the general trend of AAPL stock prices. However, like all models, it has limitations and should not be solely relied upon for financial decisions.

Sample Prediction Plot:
![__results___15_1](https://github.com/user-attachments/assets/4bca7e96-4306-4966-b872-fc0bc63e7a46)
