import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import time

# --- UI Section ---
st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.title("📈 Quantitative Stock Market Prediction App")
st.write("Compare multiple ML models on live stock data.")

# Sidebar inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", value=date.today())

# Model selection
st.sidebar.header("Model Selection")
use_lr = st.sidebar.checkbox("Linear Regression", value=True)
use_rf = st.sidebar.checkbox("Random Forest", value=True)
use_gb = st.sidebar.checkbox("Gradient Boosting", value=True)

# --- Fetch and prepare data ---
@st.cache_data
def load_data(ticker, start, end):
    try:
        # Add delay to avoid rate limiting
        time.sleep(1)
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            return None, "No data found for the given ticker and date range."
        return data, None
    except Exception as e:
        return None, f"Error downloading data: {str(e)}"

def create_features(df):
    """Create technical indicators with proper error handling"""
    df = df.copy()
    
    try:
        # Simple moving averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Price changes
        df['Price_Change_1'] = df['Close'].pct_change(1)
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['Price_Change_10'] = df['Close'].pct_change(10)
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        
        # High-Low features
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Range'] = (df['Close'] - df['Open']) / df['Open']
        
    except Exception as e:
        st.warning(f"Some features couldn't be created: {str(e)}")
    
    return df

if ticker:
    with st.spinner("Downloading stock data..."):
        data, error = load_data(ticker, start_date, end_date)
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Try using a different ticker or wait a few minutes before trying again.")
    else:
        st.subheader(f"📊 Data for {ticker} from {start_date} to {end_date}")
        
        # Check if data is valid
        if data is None or len(data) == 0:
            st.error("❌ No data available after download.")
            st.stop()
        
        # Safe data access
        try:
            current_price = data['Close'].iloc[-1]
            initial_price = data['Close'].iloc[0]
            current_price_float = float(current_price)
            initial_price_float = float(initial_price)
        except (KeyError, IndexError) as e:
            st.error(f"❌ Error accessing price data: {str(e)}")
            st.stop()
        
        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price_float:.2f}")
        with col2:
            change = current_price_float - initial_price_float
            st.metric("Total Change", f"${change:.2f}")
        with col3:
            pct_change = (change / initial_price_float) * 100
            st.metric("Total Return", f"{pct_change:.2f}%")
        with col4:
            st.metric("Data Points", len(data))

        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                               mode='lines', name='Close Price',
                               line=dict(color='#1f77b4')))
        fig.update_layout(title=f"{ticker} Stock Price",
                         xaxis_title="Date",
                         yaxis_title="Price ($)",
                         height=400)
        st.plotly_chart(fig, use_container_width=True)

        # --- Prepare features ---
        st.subheader("🔧 Feature Engineering")
        
        with st.spinner("Creating technical indicators..."):
            df = create_features(data)
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 50:
            st.error("❌ Not enough data after feature engineering. Try a wider date range.")
            st.stop()
        
        # Feature selection (simpler set to avoid issues)
        feature_columns = [
            'SMA_10', 'SMA_30', 'RSI', 
            'Price_Change_1', 'Price_Change_5', 'Price_Change_10',
            'Volume_Change', 'Volume_SMA',
            'High_Low_Range', 'Close_Open_Range'
        ]
        
        # Only use features that exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) < 5:
            st.error("❌ Not enough features available for training.")
            st.stop()
        
        # Target variable: Next day's closing price
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        X = df[available_features]
        y = df['Target']
        
        # Ensure proper data types
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        
        # Display features with shape info
        st.write(f"✅ Using {len(available_features)} technical indicators")
        st.write(f"✅ {len(X)} samples available for training")
        st.write(f"✅ X shape: {X.shape}, y shape: {y.shape}")
        
        # Feature correlation
        corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
        st.write("Top features by correlation with target:")
        st.write(corr_with_target.head(6))

        # --- Split and scale data ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Additional validation after split
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            st.error("❌ Train/test split resulted in empty datasets.")
            st.stop()
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to proper numpy arrays
        X_train_scaled = np.array(X_train_scaled, dtype=np.float64)
        X_test_scaled = np.array(X_test_scaled, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.float64)

        # --- Model Training and Evaluation ---
        st.subheader("🤖 Model Training & Evaluation")
        
        models = {}
        predictions = {}
        scores = {}
        
        # Linear Regression
        if use_lr:
            with st.spinner("Training Linear Regression..."):
                try:
                    lr = LinearRegression()
                    lr.fit(X_train_scaled, y_train)
                    pred_lr = lr.predict(X_test_scaled)
                    models['Linear Regression'] = lr
                    predictions['Linear Regression'] = pred_lr
                    scores['Linear Regression'] = {
                        'R²': r2_score(y_test, pred_lr),
                        'RMSE': np.sqrt(mean_squared_error(y_test, pred_lr))
                    }
                    st.success("✅ Linear Regression trained successfully")
                except Exception as e:
                    st.error(f"❌ Linear Regression failed: {str(e)}")
        
        # Random Forest
        if use_rf:
            with st.spinner("Training Random Forest..."):
                try:
                    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    rf.fit(X_train_scaled, y_train)
                    pred_rf = rf.predict(X_test_scaled)
                    models['Random Forest'] = rf
                    predictions['Random Forest'] = pred_rf
                    scores['Random Forest'] = {
                        'R²': r2_score(y_test, pred_rf),
                        'RMSE': np.sqrt(mean_squared_error(y_test, pred_rf))
                    }
                    st.success("✅ Random Forest trained successfully")
                except Exception as e:
                    st.error(f"❌ Random Forest failed: {str(e)}")
        
        # Gradient Boosting
        if use_gb:
            with st.spinner("Training Gradient Boosting..."):
                try:
                    gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
                    gb.fit(X_train_scaled, y_train)
                    pred_gb = gb.predict(X_test_scaled)
                    models['Gradient Boosting'] = gb
                    predictions['Gradient Boosting'] = pred_gb
                    scores['Gradient Boosting'] = {
                        'R²': r2_score(y_test, pred_gb),
                        'RMSE': np.sqrt(mean_squared_error(y_test, pred_gb))
                    }
                    st.success("✅ Gradient Boosting trained successfully")
                except Exception as e:
                    st.error(f"❌ Gradient Boosting failed: {str(e)}")
        
        if not models:
            st.error("❌ No models were successfully trained. Please check your data and model selection.")
            st.stop()
        
        # --- Results Display ---
        st.subheader("📊 Model Performance Comparison")
        
        # Performance table
        results_df = pd.DataFrame(scores).T
        results_df['R²'] = results_df['R²'].round(4)
        results_df['RMSE'] = results_df['RMSE'].round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance Metrics:**")
            st.dataframe(results_df.style.highlight_max(axis=0, subset=['R²']) \
                                   .highlight_min(axis=0, subset=['RMSE']))
        
        with col2:
            best_model = results_df['R²'].idxmax()
            st.metric("🏆 Best Model", best_model)
            st.metric("Best R² Score", f"{results_df.loc[best_model, 'R²']:.4f}")
            st.metric("Best RMSE", f"${results_df.loc[best_model, 'RMSE']:.2f}")
        
        # --- Prediction Visualization ---
        st.subheader("📈 Actual vs Predicted Prices")
        
        # Create prediction plot
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=df.index[-len(y_test):],
            y=y_test,
            mode='lines',
            name='Actual Prices',
            line=dict(color='black', width=2)
        ))
        
        # Add predictions for each model
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=df.index[-len(y_test):],
                y=pred,
                mode='lines',
                name=f'{model_name} Predictions',
                line=dict(color=colors[i % len(colors)], width=1.5)
            ))
        
        fig.update_layout(
            title="Model Predictions vs Actual Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Feature Importance ---
        st.subheader("🔍 Feature Importance")
        
        if 'Random Forest' in models:
            feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': models['Random Forest'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_importance = go.Figure(go.Bar(
                x=feature_importance['importance'],
                y=feature_importance['feature'],
                orientation='h'
            ))
            
            fig_importance.update_layout(
                title="Random Forest Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=400
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # --- Next Day Prediction ---
        st.subheader("🔮 Next Day Prediction")
        
        try:
            # Use the most recent data for prediction
            latest_features = df[available_features].iloc[-1:].values
            latest_features_scaled = scaler.transform(latest_features)
            
            next_day_predictions = {}
            for model_name, model in models.items():
                pred = model.predict(latest_features_scaled)[0]
                next_day_predictions[model_name] = float(pred)
            
            current_price_pred = float(df['Close'].iloc[-1])
            
            pred_cols = st.columns(len(next_day_predictions))
            for i, (model_name, pred_price) in enumerate(next_day_predictions.items()):
                with pred_cols[i]:
                    change = pred_price - current_price_pred
                    pct_change = (change / current_price_pred) * 100
                    
                    st.metric(
                        label=model_name,
                        value=f"${pred_price:.2f}",
                        delta=f"{change:.2f} ({pct_change:.2f}%)"
                    )
        except Exception as e:
            st.warning(f"Could not generate next day prediction: {str(e)}")

else:
    st.info("👈 Please enter a stock ticker symbol in the sidebar to get started.")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tips:**\n"
    "- Try popular tickers like AAPL, TSLA, GOOGL, MSFT\n"
    "- Use longer date ranges for better results\n"
    "- If you get rate limited, wait a few minutes and try again\n"
    "\n"
    "⚠️ **Disclaimer:** This is for educational purposes only.\n"
    "Past performance is not indicative of future results."
)
