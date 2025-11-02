import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- UI Section ---
st.title("📈 Stock Market Prediction App")
st.write("Compare multiple ML models on live stock data.")

# Sidebar inputs
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=date(2020,1,1))
end_date   = st.sidebar.date_input("End date", value=date.today())

# --- Fetch data ---
if ticker:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.warning("No data found. Check the ticker symbol or date range.")
    else:
        st.subheader(f"Data for {ticker} from {start_date} to {end_date}")
        st.line_chart(data['Close'])  # Plot closing price

        # --- Prepare data for ML ---
        close_prices = data['Close'].values
        N = 10  # Use last 10 days to predict next
        X, y = [], []
        for i in range(len(close_prices) - N):
            X.append(close_prices[i:i+N])
            y.append(close_prices[i+N])
        X, y = np.array(X), np.array(y)

        # ✅ Check: enough samples to train?
        if len(X) < 30:
            st.error("❌ Not enough data to train models. Try selecting a longer date range.")
            st.stop()

        # --- Train/test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
        
        # ✅ Extra safety check after split
        if len(X_train) == 0 or len(y_train) == 0:
            st.error("❌ Not enough training data after split. Try a longer date range or different stock.")
            st.stop()


        # --- Model 1: Linear Regression ---
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        r2_lr = r2_score(y_test, pred_lr)

        # --- Model 2: Random Forest ---
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        r2_rf = r2_score(y_test, pred_rf)

        # --- Model 3: Gradient Boosting ---
        gb = GradientBoostingRegressor(random_state=42)
        gb.fit(X_train, y_train)
        pred_gb = gb.predict(X_test)
        r2_gb = r2_score(y_test, pred_gb)

        # --- Model 4: LSTM (Neural Network) ---
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_lstm, y_train, epochs=5, batch_size=16, verbose=0)
        pred_lstm = model.predict(X_test_lstm).flatten()
        r2_lstm = r2_score(y_test, pred_lstm)

        # --- Results Table ---
        st.subheader("Model Performance (R² scores)")
        scores_df = pd.DataFrame({
            "Model": ["Linear Reg", "Random Forest", "GradientBoosting", "LSTM"],
            "R² Score": [r2_lr, r2_rf, r2_gb, r2_lstm]
        })
        st.table(scores_df)

        # --- Predictions Line Chart ---
        st.subheader("Actual vs Predicted (test set)")
        results = pd.DataFrame({
            "Actual": y_test,
            "LR":   pred_lr,
            "RF":   pred_rf,
            "GB":   pred_gb,
            "LSTM": pred_lstm
        })
        st.line_chart(results)
