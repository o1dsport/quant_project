import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

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
        data = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if data.empty:
            return None, "No data found for the given ticker and date range."
        return data, None
    except Exception as e:
        return None, f"Error downloading data: {str(e)}"

if ticker:
    data, error = load_data(ticker, start_date, end_date)
    
    if error:
        st.error(f"❌ {error}")
    else:
        st.subheader(f"📊 Data for {ticker} from {start_date} to {end_date}")
        
        # ✅ FIX: Check if data is valid before accessing
        if data is None or len(data) == 0:
            st.error("❌ No data available after download.")
            st.stop()
        
        # ✅ FIX: Safe data access
        try:
            current_price = data['Close'].iloc[-1]
            initial_price = data['Close'].iloc[0]
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

        # --- Prepare features ---
        st.subheader("🔧 Feature Engineering")
        
        # Create technical indicators
        df = data.copy()
        
        # Simple moving averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Price momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        df['Price_Rate_Of_Change'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Rate_Of_Change'] = (df['Volume'] - df['Volume'].shift(5)) / df['Volume'].shift(5)
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 50:
            st.error("❌ Not enough data after feature engineering. Try a wider date range.")
            st.stop()
        
        # Feature selection
        feature_columns = ['SMA_10', 'SMA_30', 'RSI', 'MACD', 'MACD_Signal', 
                          'BB_Upper', 'BB_Lower', 'BB_Middle', 'Momentum', 
                          'Price_Rate_Of_Change', 'Volume_SMA', 'Volume_Rate_Of_Change']
        
        # Target variable: Next day's closing price
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        X = df[feature_columns]
        y = df['Target']
        
        # ✅ CRITICAL FIX: Force proper data types and shapes
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        
        # Ensure proper shapes
        if X.ndim != 2:
            st.error(f"❌ X should be 2D but got shape {X.shape}")
            st.stop()
        if y.ndim != 1:
            st.error(f"❌ y should be 1D but got shape {y.shape}")
            st.stop()
        
        # Display features with shape info
        st.write(f"✅ Using {len(feature_columns)} technical indicators")
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
        
        # ✅ Additional validation after split
        st.write(f"After split - X_train: {X_train.shape}, y_train: {y_train.shape}")
        
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
                    # Final shape check before fitting
                    if X_train_scaled.ndim != 2 or y_train.ndim != 1:
                        st.error(f"❌ Final shape check failed: X_train_scaled {X_train_scaled.shape}, y_train {y_train.shape}")
                    else:
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
                    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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
                    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
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
                'feature': feature_columns,
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
        
        # Use the most recent data for prediction
        latest_features = df[feature_columns].iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        
        next_day_predictions = {}
        for model_name, model in models.items():
            next_day_predictions[model_name] = model.predict(latest_features_scaled)[0]
        
        current_price = df['Close'].iloc[-1]
        
        pred_cols = st.columns(len(next_day_predictions))
        for i, (model_name, pred_price) in enumerate(next_day_predictions.items()):
            with pred_cols[i]:
                change = pred_price - current_price
                pct_change = (change / current_price) * 100
                
                st.metric(
                    label=model_name,
                    value=f"${pred_price:.2f}",
                    delta=f"{change:.2f} ({pct_change:.2f}%)"
                )

else:
    st.info("👈 Please enter a stock ticker symbol in the sidebar to get started.")

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses machine learning models to predict stock prices. "
    "Past performance is not indicative of future results. "
    "Use for educational purposes only."
)
