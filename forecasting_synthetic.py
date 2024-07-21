import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to generate synthetic data
def generate_synthetic_data(start_date, periods, trend, seasonality, noise_level):
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate trend
    if trend == 'No Trend':
        y = np.zeros(periods)
    elif trend == 'Linear':
        y = np.linspace(0, 100, periods)
    elif trend == 'Exponential':
        y = np.exp(np.linspace(0, 4, periods)) - 1
    elif trend == 'Logarithmic':
        y = np.log(np.linspace(1, 100, periods))
    
    # Add seasonality
    if 'Weekly' in seasonality:
        y += 10 * np.sin(2 * np.pi * np.arange(periods) / 7)
    if 'Monthly' in seasonality:
        y += 20 * np.sin(2 * np.pi * np.arange(periods) / 30)
    if 'Yearly' in seasonality:
        y += 50 * np.sin(2 * np.pi * np.arange(periods) / 365)
    
    # Add noise
    y += np.random.normal(0, noise_level, periods)
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    return df

# Function to create forecast plot
def create_forecast_plot(df, forecast, algorithm):
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Actual vs Forecast'])
    
    # Plot actual data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Actual', marker=dict(color='blue')))
    
    # Plot forecast
    if algorithm == 'Prophet':
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', name='Upper Bound'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='Lower Bound'))
    else:
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
    
    fig.update_layout(title=f'Time Series Forecast using {algorithm}', xaxis_title='Date', yaxis_title='Value', height=600)
    return fig

# Function to perform forecasting based on selected algorithm
def forecast_algorithm(df, algorithm, forecast_periods):
    train = df['y']
    
    if algorithm == 'Prophet':
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        return forecast
    
    elif algorithm == 'ARIMA':
        model = ARIMA(train, order=(1,1,1))
        results = model.fit()
        forecast = results.forecast(steps=forecast_periods)
    elif algorithm == 'ETS':
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7)
        results = model.fit()
        forecast = results.forecast(forecast_periods)
    elif algorithm == 'SARIMA':
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7))
        results = model.fit()
        forecast = results.forecast(steps=forecast_periods)
    elif algorithm == 'Simple Moving Average':
        window_size = min(30, len(train) // 2)
        ma = train.rolling(window=window_size).mean()
        last_ma = ma.iloc[-1]
        forecast = pd.Series([last_ma] * forecast_periods)
    
    # For non-Prophet algorithms, combine the original data with the forecast
    full_dates = pd.date_range(start=df['ds'].iloc[0], end=df['ds'].iloc[-1] + pd.Timedelta(days=forecast_periods), freq='D')
    full_forecast = pd.Series(index=full_dates)
    full_forecast.loc[df['ds']] = train
    
    # Ensure forecast index is compatible with full_forecast index
    forecast_index = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
    forecast = pd.Series(forecast.values, index=forecast_index)
    
    full_forecast.loc[forecast.index] = forecast
    full_forecast = full_forecast.interpolate()
    
    return full_forecast

# Streamlit app
st.title('Time Series Forecasting with Synthetic Data')

# Sidebar inputs
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2023-01-01'))
periods = st.sidebar.slider('Number of Periods', min_value=100, max_value=1000, value=365)

trend_options = ['No Trend', 'Linear', 'Exponential', 'Logarithmic']
trend = st.sidebar.selectbox('Trend', trend_options)

seasonality_options = ['No Seasonality', 'Weekly', 'Monthly', 'Yearly', 'Weekly + Monthly', 'Weekly + Yearly', 'Monthly + Yearly', 'Weekly + Monthly + Yearly']
seasonality = st.sidebar.selectbox('Seasonality', seasonality_options)

noise_level = st.sidebar.slider('Noise Level', min_value=0.1, max_value=10.0, value=1.0)
forecast_periods = st.sidebar.slider('Forecast Periods', min_value=30, max_value=365, value=90)

# Algorithm selection
algorithm_options = ['Prophet', 'ARIMA', 'ETS', 'SARIMA', 'Simple Moving Average']
algorithm = st.sidebar.selectbox('Forecasting Algorithm', algorithm_options)

# Generate synthetic data
df = generate_synthetic_data(start_date, periods, trend, seasonality, noise_level)

# Perform forecasting
forecast = forecast_algorithm(df, algorithm, forecast_periods)

# Create the plot
fig = create_forecast_plot(df, forecast, algorithm)
st.plotly_chart(fig)

# Display key metrics
st.subheader('Key Metrics')
if algorithm == 'Prophet':
    train_forecast = forecast[forecast['ds'].isin(df['ds'])]
    mae = mean_absolute_error(df['y'], train_forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(df['y'], train_forecast['yhat']))
else:
    # Align the forecast with the original data
    aligned_forecast = forecast.reindex(df['ds'], method='nearest')
    aligned_forecast = aligned_forecast.dropna()
    
    # Use only the overlapping period for metric calculation
    common_dates = df['ds'][df['ds'].isin(aligned_forecast.index)]
    
    if len(common_dates) > 0:
        actual_values = df.loc[df['ds'].isin(common_dates), 'y']
        forecast_values = aligned_forecast[common_dates]
        
        mae = mean_absolute_error(actual_values, forecast_values)
        rmse = np.sqrt(mean_squared_error(actual_values, forecast_values))
    else:
        mae = np.nan
        rmse = np.nan

if np.isnan(mae) or np.isnan(rmse):
    st.write("Unable to calculate metrics. No overlap between actual data and forecast.")
else:
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Square Error (RMSE): {rmse:.2f}")

# Display raw data
st.subheader('Raw Data')
st.write(df)