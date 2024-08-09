import pandas as pd
import yfinance as yf
import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import config  
import ta 

def collect_stock_data_yahoo(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            print("No data found for the symbol.")
        else:
            print("Data fetched from Yahoo Finance:")
        return df
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()

def store_data_in_mysql(df, symbol):
    connection = mysql.connector.connect(**config.DATABASE_CONFIG)
    cursor = connection.cursor()
    
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {symbol} (
            Date DATE PRIMARY KEY,
            Open FLOAT,
            High FLOAT,
            Low FLOAT,
            Close FLOAT,
            Volume INT
        )
    ''')
    
   
    df.reset_index(inplace=True)
    
    df.dropna(inplace=True) 
    
    
    for i, row in df.iterrows():
        cursor.execute(f'''
            INSERT INTO {symbol} (Date, Open, High, Low, Close, Volume)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            Open = VALUES(Open),
            High = VALUES(High),
            Low = VALUES(Low),
            Close = VALUES(Close),
            Volume = VALUES(Volume)
        ''', (row['Date'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
    
    connection.commit()
    cursor.close()
    connection.close()

def load_data_from_mysql(symbol):
    connection = mysql.connector.connect(**config.DATABASE_CONFIG)
    query = f'SELECT * FROM {symbol}'
    df = pd.read_sql(query, connection)
    connection.close()
    return df

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.ffill() 

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    
    df.dropna(inplace=True)
    
    if df.empty:
        print("DataFrame is empty after preprocessing.")
        return np.array([]), np.array([]), None
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(model, X_test, y_test, scaler):
    predicted_prices = model.predict(X_test)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    r2 = r2_score(actual_prices, predicted_prices)
    
    return rmse, r2, actual_prices, predicted_prices


if __name__ == "__main__":
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-08-09'
    

    yahoo_data = collect_stock_data_yahoo(symbol, start_date, end_date)
    if not yahoo_data.empty:
        store_data_in_mysql(yahoo_data, symbol)
    

    df = load_data_from_mysql(symbol)
    if df.empty:
        print('DataFrame is empty. No data found for the symbol.')
    else:
        print("Data loaded from MySQL:")
        X, y, scaler = preprocess_data(df)
        
        if X.size == 0 or y.size == 0:
            print("No data available for training the model. Exiting...")
        else:
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            model = build_model((X_train.shape[1], 1))
            
            early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
            checkpoint = ModelCheckpoint('best_model.keras', monitor='loss', verbose=1, save_best_only=True, mode='min')
            
            model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping, checkpoint])
            
            rmse, r2, actual_prices, predicted_prices = evaluate_model(model, X_test, y_test, scaler)
            
            print(f'RMSE: {rmse:.2f}')
            print(f'R-squared: {r2:.2f}')

            plt.figure(figsize=(12, 6))
            plt.plot(actual_prices, label='Actual Price')
            plt.plot(predicted_prices, label='Predicted Price')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title(f'{symbol} Stock Price Prediction')
            plt.show()
