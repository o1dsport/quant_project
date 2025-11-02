import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Stock Predictor", layout="wide")

# --- UI ---
st.title("📈 Stock Market Prediction App")
st.write("Compare multiple ML models on live stock data.")

ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", value=date.today())

if ticker:
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.warning("No data found. Check the ticker or range.")
        st.stop()

    st.subheader(f"{ticker} Closing Prices")
    st.line_chart(data['Close'])

    close_prices = data['Close'].values
    N = 10
    X, y = [], []

    for i in range(len(close_prices) - N):
        X.append(close_prices[i:i+N])
        y.append(close_prices[i+N])

    X = np.array(X)
    y = np.array(y)

    # SHAPE CHECKS (CRITICAL)
    st.write(f"✅ Raw X shape: {X.shape}")
    st.write(f"✅ Raw y shape: {y.shape}")

    if X.ndim != 2:
        X = X.reshape(-1, N)
        st.write("🔄 Reshaped X to 2D.")

    if y.ndim != 1:
        y = y.flatten()
        st.write("🔄 Flattened y to 1D.")

    if X.shape[0] != y.shape[0] or X.shape[0] < 30:
        st.error("❌ Data size mismatch or not enough samples.")
        st.stop()

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    st.write(f"📊 X_train shape: {X_train.shape}")
    st.write(f"📊 y_train shape: {y_train.shape}")

    if X_train.ndim != 2 or y_train.ndim != 1:
        st.error("❌ Training shape error: X must be 2D and y must be 1D.")
        st.stop()

    # --- Linear Regression ---
    lr = LinearRegression()
    st.code(f"X_train shape: {X_train.shape}, ndim: {X_train.ndim}")
    st.code(f"y_train shape: {y_train.shape}, ndim: {y_train.ndim}")

    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    r2_lr = r2_score(y_test, pred_lr)

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, pred_rf)

    # --- Gradient Boosting ---
    gb = GradientBoostingRegressor()
    gb.fit(X_train, y_train)
    pred_gb = gb.predict(X_test)
    r2_gb = r2_score(y_test, pred_gb)

    # --- LSTM ---
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train, epochs=5, batch_size=16, verbose=0)
    pred_lstm = model.predict(X_test_lstm).flatten()
    r2_lstm = r2_score(y_test, pred_lstm)

    # --- Results ---
    st.subheader("📊 Model R² Scores")
    st.table(pd.DataFrame({
        "Model": ["LinearReg", "RandomForest", "GradientBoost", "LSTM"],
        "R²": [r2_lr, r2_rf, r2_gb, r2_lstm]
    }))

    st.subheader("📈 Actual vs Predicted (Test Set)")
    st.line_chart(pd.DataFrame({
        "Actual": y_test,
        "LinearReg": pred_lr,
        "RandomForest": pred_rf,
        "GradientBoost": pred_gb,
        "LSTM": pred_lstm
    }))
