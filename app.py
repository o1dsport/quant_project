# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# For LSTM model (Keras)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- UI: Title and sidebar inputs ---
st.title("📈 Stock Market Prediction App")
st.write("Compare multiple ML models on live stock data.")

# Sidebar: user selects stock ticker and date range
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=date(2020,1,1))
end_date   = st.sidebar.date_input("End date",   value=date.today())

# --- Fetch live stock data with yfinance ---
if ticker:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.warning("No data found. Check the ticker symbol or date range.")
    else:
        st.subheader(f"Data for {ticker} from {start_date} to {end_date}")
        st.line_chart(data['Close'])  # plot closing price

        # --- Prepare data for ML models ---
        # Use closing prices and create sliding-window features
        close_prices = data['Close'].values
        N = 10  # use past 10 days to predict next day
        X, y = [], []
        for i in range(len(close_prices) - N):
            X.append(close_prices[i:i+N])
            y.append(close_prices[i+N])
        X, y = np.array(X), np.array(y)

        # Ensure we have enough data for training
        if len(X) > 10:

            # Split into train/test sets (80/20 split)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

            # --- Train and evaluate models ---
            # 1. Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            pred_lr = lr.predict(X_test)
            r2_lr = r2_score(y_test, pred_lr)

            # 2. Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            pred_rf = rf.predict(X_test)
            r2_rf = r2_score(y_test, pred_rf)

            # 3. Gradient Boosting
            gb = GradientBoostingRegressor(random_state=42)
            gb.fit(X_train, y_train)
            pred_gb = gb.predict(X_test)
            r2_gb = r2_score(y_test, pred_gb)

            # 4. LSTM (Deep Learning)
            # Reshape data for LSTM: (samples, timesteps, features)
            X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test_lstm  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))
            model = Sequential([
                LSTM(50, input_shape=(X_train_lstm.shape[1], 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_lstm, y_train, epochs=5, batch_size=16, verbose=0)  # small epochs for demo
            pred_lstm = model.predict(X_test_lstm).flatten()
            r2_lstm = r2_score(y_test, pred_lstm)

            # --- Display model performance ---
            st.subheader("Model Performance (R² scores)")
            scores_df = pd.DataFrame({
                "Model": ["Linear Reg", "Random Forest", "GradientBoosting", "LSTM"],
                "R² Score": [r2_lr, r2_rf, r2_gb, r2_lstm]
            })
            st.table(scores_df)

            # --- Show predictions vs actual ---
            st.subheader("Actual vs Predicted (test set)")
            results = pd.DataFrame({
                "Actual": y_test,
                "LR":   pred_lr,
                "RF":   pred_rf,
                "GB":   pred_gb,
                "LSTM": pred_lstm
            })
            st.line_chart(results)  # line chart of actual and predictions
        else:
            st.warning("Not enough data to train models.")
