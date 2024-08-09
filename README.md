# Stock_prediction_model

Stock Price Prediction using LSTM
This project aims to predict stock prices using an LSTM (Long Short-Term Memory) neural network. It fetches historical stock data from Yahoo Finance, preprocesses it, and trains an LSTM model to forecast future stock prices.
Features

    Fetches stock data from Yahoo Finance using the yfinance library
    Stores the data in a MySQL database using the mysql.connector library
    Preprocesses the data by calculating technical indicators like Moving Average (MA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD)
    Scales the data using Min-Max Scaler from sklearn.preprocessing
    Splits the data into training and testing sets
    Builds an LSTM model using the keras library
    Trains the model using early stopping and model checkpointing
    Evaluates the model's performance using RMSE (Root Mean Squared Error) and R-squared metrics from sklearn.metrics
    Visualizes the actual vs. predicted prices using matplotlib.pyplot

Dependencies
The project requires the following dependencies:

    pandas
    yfinance
    mysql.connector
    numpy
    matplotlib
    scikit-learn
    keras
    ta
