# Stock_prediction_model

Stock Price Prediction using LSTM
This project aims to predict stock prices using an LSTM (Long Short-Term Memory) neural network. It fetches historical stock data from Yahoo Finance, preprocesses it, and trains an LSTM model to forecast future stock prices.
Features

    - Fetches stock data from Yahoo Finance using the yfinance library
    - Stores the data in a MySQL database using the mysql.connector library
    - Preprocesses the data by calculating technical indicators like Moving Average (MA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD)
    - Scales the data using Min-Max Scaler from sklearn.preprocessing
    - Splits the data into training and testing sets
    - Builds an LSTM model using the keras library
    - Trains the model using early stopping and model checkpointing
    - Evaluates the model's performance using RMSE (Root Mean Squared Error) and R-squared metrics from sklearn.metrics
    - Visualizes the actual vs. predicted prices using matplotlib.pyplot

### Dependencies
The project requires the following dependencies:

    - pandas
    - yfinance
    - mysql.connector
    - numpy
    - matplotlib
    - scikit-learn
    - keras
    - ta

### Results
The LSTM model achieved an RMSE of 4.30 and an R-squared value of 0.95 on the test set. This indicates that the model is able to explain 95% of the variance in the predicted stock prices and has an average error of 4.30 units.

### Visualization of Actual vs. Predicted Prices

The model's predictions are visualized against the actual stock prices. The plot below shows how closely the predicted prices follow the actual prices over the training period.

![Actual vs. Predicted Prices](path_to_your_image.png)


#### Usage

-Clone the repository to your local machine.
-Create a config.py file in your project directory and fill in your MySQL database credentials and Alpha Vantage API key.
-Install the required dependencies using pip install -r requirements.txt.
-Run the main.py script to fetch data, train the model, and visualize the results.

