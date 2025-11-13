import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Stock Market Predictor", layout="wide")
st.title("Stock Market Prediction")
st.markdown("Compare **Linear Regression**, **Gradient Boosting**, and **LSTM** models on real market data.")


def directional_accuracy(y_true, y_pred):
    """Compute direction (up/down) match % and confusion matrix safely"""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return 0.0, np.zeros((2, 2), dtype=int)
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    acc = np.sum(actual_dir == pred_dir) / len(actual_dir) * 100
    cm = confusion_matrix(actual_dir > 0, pred_dir > 0, labels=[False, True])
    return acc, cm


symbol = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, INFY.NS):", "RELIANCE.NS")
start_date = st.date_input("Start Date", date(2015, 1, 1))
end_date = st.date_input("End Date", date.today())
future_days = st.slider("Forecast next N days", 1, 30, 4)

if st.button("Train & Predict"):
    with st.spinner("Downloading data and training models..."):
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found for that symbol.")
            st.stop()

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)

        X = df[['Open', 'High', 'Low', 'Volume', 'Return']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_mse = mean_squared_error(y_test, lr_preds)
        lr_r2 = r2_score(y_test, lr_preds)
        lr_acc, lr_cm = directional_accuracy(y_test, lr_preds)


        gb = GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42
        )
        gb.fit(X_train, y_train)
        gb_preds = gb.predict(X_test)
        gb_mse = mean_squared_error(y_test, gb_preds)
        gb_r2 = r2_score(y_test, gb_preds)
        gb_acc, gb_cm = directional_accuracy(y_test, gb_preds)


        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['Close']])

        lookback = 60
        X_lstm, y_lstm = [], []
        for i in range(lookback, len(scaled)):
            X_lstm.append(scaled[i-lookback:i])
            y_lstm.append(scaled[i])
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

        split = int(len(X_lstm) * 0.8)
        X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
        y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=0, callbacks=[es])

        lstm_preds = model.predict(X_test_lstm)
        lstm_preds = scaler.inverse_transform(lstm_preds)
        actual_prices = scaler.inverse_transform(y_test_lstm)

        lstm_mse = mean_squared_error(actual_prices, lstm_preds)
        lstm_r2 = r2_score(actual_prices, lstm_preds)
        lstm_acc, lstm_cm = directional_accuracy(actual_prices, lstm_preds)


        st.subheader("Model Performance Metrics")
        # R² can be negative on test data; clip at 0 for display to avoid confusion
        r2_display = [max(lr_r2, 0.0), max(gb_r2, 0.0), max(lstm_r2, 0.0)]
        metrics = pd.DataFrame({
            "Model": ["Linear Regression", "Gradient Boosting", "LSTM"],
            "MSE": [lr_mse, gb_mse, lstm_mse],
            "R² Score (clipped ≥ 0)": r2_display,
            "Directional Accuracy (%)": [lr_acc, gb_acc, lstm_acc]
        })

        st.dataframe(metrics.style.highlight_max(
            subset=["R² Score (clipped ≥ 0)", "Directional Accuracy (%)"], color="lightgreen"))



        st.subheader("Actual vs Predicted Prices (Test Set)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test.values, label="Actual", color='black', linewidth=2)
        ax.plot(lr_preds, label="Linear Regression", color='blue')
        ax.plot(gb_preds, label="Gradient Boosting", color='orange')
        ax.plot(np.linspace(0, len(y_test), len(lstm_preds)), lstm_preds, label="LSTM", color='green')
        ax.legend()
        st.pyplot(fig)


        st.subheader("Confusion Matrices (Up/Down Prediction)")
        col1, col2, col3 = st.columns(3)
        for c, cm, name, cmap in zip(
            [col1, col2, col3],
            [lr_cm, gb_cm, lstm_cm],
            ["Linear Regression", "Gradient Boosting", "LSTM"],
            ["Blues", "Oranges", "Greens"]
        ):
            with c:
                st.write(f"**{name}**")
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)


        st.subheader(f"{symbol} - Future {future_days}-Day Forecast")
        last_seq = scaled[-lookback:]
        future_preds = []
        seq = last_seq.copy()

        for _ in range(future_days):
            pred = model.predict(seq.reshape(1, lookback, 1))
            future_preds.append(pred[0][0])
            seq = np.append(seq[1:], pred, axis=0)

        future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(future_days)]
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_prices.flatten()})
        st.dataframe(forecast_df)

        fig_fut, ax_fut = plt.subplots(figsize=(10, 6))
        ax_fut.plot(df.index[-100:], df["Close"].tail(100), label="Recent Actual", color="blue")
        ax_fut.plot(future_dates, future_prices, label="Forecast", color="red", marker='o')
        ax_fut.legend()
        st.pyplot(fig_fut)

        st.success("Models trained, evaluated, and forecast completed successfully!")
