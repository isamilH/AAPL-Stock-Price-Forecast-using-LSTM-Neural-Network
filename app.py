import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Function to load and preprocess data
def load_data(stock, start, end):
    stock_data = yf.download(stock, start=start, end=end)
    df = stock_data[['Close']].reset_index()
    df.columns = ['Date', 'Price']
    df.set_index('Date', inplace=True)
    df = df.resample('D').ffill()
    return df


# Function to scale data and create sequences
def create_sequences(data, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i - time_steps:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(X), np.array(y), scaler


# Build and compile the model
def build_model(time_steps):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Streamlit App Interface
st.title("Stock Price Prediction App")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
start_date = st.date_input("Start Date:", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date:", pd.to_datetime("2024-10-02"))
predict_button = st.button("Predict")

if predict_button:
    with st.spinner("Loading data and training model..."):
        # Load and prepare data
        df = load_data(stock_symbol, start_date, end_date)
        st.line_chart(df['Price'], width=700, height=400)

        # Prepare sequences
        time_steps = 60
        X, y, scaler = create_sequences(df['Price'], time_steps)
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        # Train model
        model = build_model(time_steps)
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        # Make predictions on latest data
        recent_data = df['Price'].values[-time_steps:].reshape(-1, 1)
        recent_data_scaled = scaler.transform(recent_data)
        X_input = recent_data_scaled.reshape(1, time_steps, 1)
        prediction_scaled = model.predict(X_input)
        prediction = scaler.inverse_transform(prediction_scaled)

        # Display result
        st.write(f"Predicted price for {stock_symbol} on the next trading day: ${prediction[0][0]:.2f}")
