import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import time
import requests
import io

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

# --- Alternative data sources and fallbacks ---
def get_sample_data():
    """Provide sample data when live data fails"""
    dates = pd.date_range(start='2020-01-01', end=date.today(), freq='D')
    np.random.seed(42)
    
    # Generate realistic sample data
    price = 100 + np.cumsum(np.random.normal(0, 2, len(dates)))
    volume = np.random.randint(1000000, 50000000, len(dates))
    
    sample_data = pd.DataFrame({
        'Open': price * 0.99,
        'High': price * 1.02,
        'Low': price * 0.98,
        'Close': price,
        'Volume': volume
    }, index=dates)
    
    return sample_data

def load_data_robust(ticker, start, end):
    """Robust data loading with multiple fallbacks"""
    try:
        # Try yfinance first
        time.sleep(2)  # Rate limiting protection
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        if not data.empty:
            return data, None
        
        # If yfinance fails, try Alpha Vantage (free tier)
        st.warning("📡 Trying alternative data source...")
        return load_data_alphavantage(ticker)
        
    except Exception as e:
        st.warning("🚨 Using sample data for demonstration")
        sample_data = get_sample_data()
        return sample_data, f"Live data unavailable. Using sample data. Error: {str(e)}"

def load_data_alphavantage(ticker):
    """Try Alpha Vantage as backup"""
    try:
        # You can get a free API key from https://www.alphavantage.co/support/#api-key
        api_key = "demo"  # Free demo key (limited)
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&datatype=csv"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = pd.read_csv(io.StringIO(response.text))
            if not data.empty:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
                data = data.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low', 
                    'close': 'Close', 'volume': 'Volume'
                })
                return data, None
    except:
        pass
    
    # Final fallback to sample data
    return get_sample_data(), "Using sample data for demonstration"

# --- Feature Engineering ---
def create_simple_features(df):
    """Create basic technical indicators"""
    df = df.copy()
    
    # Price-based features
    df['Returns_1'] = df['Close'].pct_change(1)
    df['Returns_5'] = df['Close'].pct_change(5)
    df['Returns_10'] = df['Close'].pct_change(10)
    
    # Rolling statistics
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility_10'] = df['Returns_1'].rolling(window=10).std()
    
    # Volume features
    df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Rate'] = df['Volume'] / df['Volume_SMA']
    
    # Price position features
    df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
    df['Close_Open_Ratio'] = (df['Close'] - df['Open']) / df['Open']
    
    return df

# --- Main Application Logic ---
if ticker:
    with st.spinner("🔄 Loading stock data..."):
        data, warning_msg = load_data_robust(ticker, start_date, end_date)
    
    if warning_msg:
        st.warning(warning_msg)
    
    st.subheader(f"📊 Data for {ticker} from {start_date} to {end_date}")
    
    # Safe data access
    try:
        current_price = float(data['Close'].iloc[-1])
        initial_price = float(data['Close'].iloc[0])
    except (KeyError, IndexError) as e:
        st.error(f"❌ Error accessing price data: {str(e)}")
        st.stop()
    
    # Display basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        change = current_price - initial_price
        st.metric("Total Change", f"${change:.2f}")
    with col3:
        pct_change = (change / initial_price) * 100
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

    # --- Feature Engineering ---
    st.subheader("🔧 Feature Engineering")
    
    with st.spinner("Creating features..."):
        df = create_simple_features(data)
    
    # Drop NaN values
    df = df.dropna()
    
    if len(df) < 30:
        st.error("❌ Not enough data for training. Try a wider date range.")
        st.stop()
    
    # Feature selection
    feature_columns = [
        'Returns_1', 'Returns_5', 'Returns_10',
        'SMA_10', 'SMA_20', 'Volatility_10',
        'Volume_Rate', 'High_Low_Ratio', 'Close_Open_Ratio'
    ]
    
    # Only use features that exist
    available_features = [col for col in feature_columns if col in df.columns]
    
    if len(available_features) < 3:
        st.error("❌ Not enough features available for training.")
        st.stop()
    
    # Target: Next day's closing price
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    
    X = df[available_features]
    y = df['Target']
    
    st.write(f"✅ Using {len(available_features)} features")
    st.write(f"✅ {len(X)} samples available")
    st.write(f"📊 Feature matrix shape: {X.shape}")

    # --- Model Training ---
    st.subheader("🤖 Model Training & Evaluation")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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
                st.success("✅ Linear Regression trained")
            except Exception as e:
                st.error(f"❌ Linear Regression failed: {str(e)}")
    
    # Random Forest
    if use_rf:
        with st.spinner("Training Random Forest..."):
            try:
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X_train_scaled, y_train)
                pred_rf = rf.predict(X_test_scaled)
                models['Random Forest'] = rf
                predictions['Random Forest'] = pred_rf
                scores['Random Forest'] = {
                    'R²': r2_score(y_test, pred_rf),
                    'RMSE': np.sqrt(mean_squared_error(y_test, pred_rf))
                }
                st.success("✅ Random Forest trained")
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
                st.success("✅ Gradient Boosting trained")
            except Exception as e:
                st.error(f"❌ Gradient Boosting failed: {str(e)}")
    
    if not models:
        st.error("❌ No models trained successfully")
        st.stop()
    
    # --- Results ---
    st.subheader("📊 Model Performance")
    
    results_df = pd.DataFrame(scores).T
    results_df['R²'] = results_df['R²'].round(4)
    results_df['RMSE'] = results_df['RMSE'].round(2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Performance Metrics:**")
        st.dataframe(results_df)
    
    with col2:
        best_model = results_df['R²'].idxmax()
        st.metric("🏆 Best Model", best_model)
        st.metric("Best R²", f"{results_df.loc[best_model, 'R²']:.4f}")
    
    # Predictions chart
    st.subheader("📈 Predictions vs Actual")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=df.index[-len(y_test):], y=y_test,
        mode='lines', name='Actual', line=dict(color='black', width=2)
    ))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (name, pred) in enumerate(predictions.items()):
        fig_pred.add_trace(go.Scatter(
            x=df.index[-len(y_test):], y=pred,
            mode='lines', name=f'{name} Pred', line=dict(color=colors[i], width=1.5)
        ))
    
    fig_pred.update_layout(height=400, title="Model Predictions")
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Feature Importance
    if 'Random Forest' in models:
        st.subheader("🔍 Feature Importance")
        importance_df = pd.DataFrame({
            'feature': available_features,
            'importance': models['Random Forest'].feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig_imp = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h'
        ))
        fig_imp.update_layout(height=300, title="Random Forest Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("👈 Enter a stock ticker to get started")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tips:**\n"
    "- Try: AAPL, TSLA, GOOGL, MSFT\n" 
    "- Use wider date ranges\n"
    "- App uses fallback data if live data fails\n"
    "\n⚠️ **Educational use only**"
)
