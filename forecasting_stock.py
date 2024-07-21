import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objs as go
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
import random

# Conditional import for auto_arima
try:
    from pmdarima import auto_arima
    auto_arima_available = True
except ImportError:
    auto_arima_available = False
    st.warning("pmdarima (auto_arima) is not available. Some functionality may be limited.")

# List of 25 US stock exchange tickers and 25 Indian NS tickers
us_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'NVDA', 'HD', 'DIS', 'BAC', 'ADBE', 'NFLX', 'CRM', 'CMCSA', 'XOM', 'VZ', 'KO', 'INTC']
indian_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'LT.NS', 'AXISBANK.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'BAJAJFINSV.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'TECHM.NS', 'HDFCLIFE.NS']

# Function to fetch stock data
def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = df['Date'].dt.tz_localize(None)
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
        df = df[['ds', 'y']].dropna()  # Remove any rows with NaN values
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Function to create forecast plot
def create_forecast_plot(df, forecast, algorithm):
    fig = go.Figure()
    
    # Plot actual data
    fig.add_trace(go.Bar(x=df['ds'], y=df['y'], name='Actual', marker_color='darkblue'))
    
    # Plot forecast
    forecast_dates = forecast.index
    forecast_values = forecast.values
    
    fig.add_trace(go.Bar(x=forecast_dates, y=forecast_values, name='Forecast', marker_color='lightblue'))
    
    fig.update_layout(
        title=f'Stock Price Forecast using {algorithm}',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        barmode='overlay'
    )
    return fig

# Function to calculate metrics
def calculate_metrics(actual, forecast, n_params=1):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, forecast)
    
    # Adjusted R-squared
    n = len(actual)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    # Theil's U statistic
    numerator = np.sqrt(np.mean((forecast - actual)**2))
    denominator = np.sqrt(np.mean(actual**2)) + np.sqrt(np.mean(forecast**2))
    theil_u = numerator / denominator
    
    # Durbin-Watson statistic
    residuals = actual - forecast
    dw = durbin_watson(residuals)
    
    return mae, rmse, mse, r2, adj_r2, mape, theil_u, dw

# Function to perform forecasting based on selected algorithm
def forecast_algorithm(df, algorithm, forecast_periods):
    train = df['y'].values
    dates = df['ds'].values
    
    if algorithm == 'ARIMA':
        model = ARIMA(train, order=(1,1,1))
        results = model.fit()
        forecast = results.forecast(steps=forecast_periods)
    
    elif algorithm == 'SARIMA':
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7))
        results = model.fit()
        forecast = results.forecast(steps=forecast_periods)
    
    elif algorithm == 'Auto ARIMA':
        if auto_arima_available:
            model = auto_arima(train, seasonal=True, m=7)
            forecast = model.predict(n_periods=forecast_periods)
        else:
            st.error("Auto ARIMA is not available. Please install pmdarima.")
            return None
    
    elif algorithm == 'Holt-Winters':
        model = ExponentialSmoothing(train, seasonal_periods=7, trend='add', seasonal='add')
        results = model.fit()
        forecast = results.forecast(steps=forecast_periods)
    
    elif algorithm == 'Theta':
        model = ThetaModel(train, period=7)
        results = model.fit()
        forecast = results.forecast(steps=forecast_periods)
    
    else:
        X = np.arange(len(train)).reshape(-1, 1)
        if algorithm == 'XGBoost':
            model = XGBRegressor(n_estimators=100)
        elif algorithm == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100)
        elif algorithm == 'SVR':
            model = SVR(kernel='rbf')
        elif algorithm == 'Linear Regression':
            model = LinearRegression()
        elif algorithm == 'Ridge':
            model = Ridge(alpha=1.0)
        elif algorithm == 'Lasso':
            model = Lasso(alpha=1.0)
        elif algorithm == 'Gradient Boosting':
            model = GradientBoostingRegressor(n_estimators=100)
        elif algorithm == 'Decision Tree':
            model = DecisionTreeRegressor()
        elif algorithm == 'KNN':
            model = KNeighborsRegressor(n_neighbors=5)
        elif algorithm == 'AdaBoost':
            model = AdaBoostRegressor(n_estimators=100)
        elif algorithm == 'ElasticNet':
            model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        
        model.fit(X, train)
        forecast = model.predict(np.arange(len(train), len(train) + forecast_periods).reshape(-1, 1))
    
    # Combine the original data with the forecast
    full_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='B')
    forecast = pd.Series(forecast, index=full_dates)
    
    return forecast

# Streamlit app
st.title('Advanced Stock Price Forecasting')

# Sidebar inputs
ticker = st.sidebar.text_input('Stock Ticker', value='AAPL')

# Add "Select Stock" button
if st.sidebar.button('Select Random Stock'):
    all_tickers = us_tickers + indian_tickers
    ticker = random.choice(all_tickers)
    st.sidebar.write(f"Randomly selected stock: {ticker}")

start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2023-11-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-12-31'))
forecast_periods = st.sidebar.slider('Forecast Periods (Days)', min_value=5, max_value=252, value=30)

# Algorithm selection
algorithm_options = ['All Algos', 'ARIMA', 'SARIMA', 'Auto ARIMA', 'XGBoost', 'Random Forest', 'SVR', 
                     'Linear Regression', 'Ridge', 'Lasso', 'Gradient Boosting', 'Holt-Winters', 'Theta',
                     'Decision Tree', 'KNN', 'AdaBoost', 'ElasticNet']
algorithm = st.sidebar.selectbox('Forecasting Algorithm', algorithm_options)

# Fetch stock data
df = get_stock_data(ticker, start_date, end_date)

if df is not None and not df.empty:
    if algorithm != 'All Algos':
        # Perform forecasting for a single algorithm
        try:
            forecast = forecast_algorithm(df, algorithm, forecast_periods)
            if forecast is not None:
                # Create the plot
                fig = create_forecast_plot(df, forecast, algorithm)
                st.plotly_chart(fig)

                # Display key metrics
                st.subheader('Key Metrics')
                actual_values = df['y'][-len(forecast):]
                forecast_values = forecast[:len(actual_values)]
                mae, rmse, mse, r2, adj_r2, mape, theil_u, dw = calculate_metrics(actual_values, forecast_values)

                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"Root Mean Square Error (RMSE): {rmse:.2f}")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"R-squared (R2) Score: {r2:.4f}")
                st.write(f"Adjusted R-squared: {adj_r2:.4f}")
                st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                st.write(f"Theil's U statistic: {theil_u:.4f}")
                st.write(f"Durbin-Watson statistic: {dw:.4f}")
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")

    else:
        # Perform forecasting for all algorithms and create a comparison table
        st.subheader('Algorithm Comparison')
        comparison_data = []

        for algo in algorithm_options[1:]:  # Skip 'All Algos'
            try:
                forecast = forecast_algorithm(df, algo, forecast_periods)
                if forecast is not None:
                    actual_values = df['y'][-len(forecast):]
                    forecast_values = forecast[:len(actual_values)]
                    mae, rmse, mse, r2, adj_r2, mape, theil_u, dw = calculate_metrics(actual_values, forecast_values)
                    
                    comparison_data.append({
                        'Algorithm': algo,
                        'MAE': mae,
                        'RMSE': rmse,
                        'MSE': mse,
                        'R2 Score': r2,
                        'Adj R2': adj_r2,
                        'MAPE': mape,
                        'Theil U': theil_u,
                        'Durbin-Watson': dw
                    })
            except Exception as e:
                st.warning(f"Error in {algo} forecasting: {str(e)}")
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Metric selection for sorting
            sort_metric = st.selectbox('Sort by Metric', ['MAE', 'RMSE', 'MSE', 'R2 Score', 'Adj R2', 'MAPE', 'Theil U', 'Durbin-Watson'])
            ascending = st.checkbox('Ascending Order', value=True)
            
            # Sort the dataframe
            comparison_df = comparison_df.sort_values(sort_metric, ascending=ascending)
            
            # Display the table
            st.table(comparison_df.style.format({
                'MAE': '{:.2f}',
                'RMSE': '{:.2f}',
                'MSE': '{:.2f}',
                'R2 Score': '{:.4f}',
                'Adj R2': '{:.4f}',
                'MAPE': '{:.2f}%',
                'Theil U': '{:.4f}',
                'Durbin-Watson': '{:.4f}'
            }))
        else:
            st.warning("No valid forecasts could be generated for any algorithm.")

    # Display raw data
    st.subheader('Raw Data')
    st.write(df)
else:
    st.error(f"No data available for {ticker} in the specified date range.")
