import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from datetime import datetime, timedelta
import random
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as ETSModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# List of 25 US stock exchange tickers and 25 Indian NS tickers
us_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'NVDA', 'HD', 'DIS', 'BAC', 'ADBE', 'NFLX', 'CRM', 'CMCSA', 'XOM', 'VZ', 'KO', 'INTC']
indian_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'LT.NS', 'AXISBANK.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'BAJAJFINSV.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'TECHM.NS', 'HDFCLIFE.NS']

@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = df['Date'].dt.tz_localize(None)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def create_forecast_plot(historical_df, forecast_df, plot_type='line'):
    fig = go.Figure()

    y_max = max(historical_df['Close'].max(), forecast_df['yhat'].max())

    if plot_type == 'line':
        fig.add_trace(go.Scatter(x=historical_df['Date'], y=historical_df['Close'], name='Historical Data', line=dict(color='blue')))
    else:
        fig.add_trace(go.Bar(x=historical_df['Date'], y=historical_df['Close'], name='Historical Data', marker_color='blue'))
    
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', line=dict(color='red')))
    
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], name='Lower Bound', line=dict(color='rgba(255,0,0,0.3)')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], name='Upper Bound', line=dict(color='rgba(255,0,0,0.3)', dash='dash')))

    fig.update_layout(
        title='Stock Price and Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis=dict(range=[0, y_max * 1.1]),
        height=600,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def prepare_data(df):
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def arima_forecast(df, periods, order):
    model = ARIMA(df['Close'], order=order)
    results = model.fit()
    forecast = results.get_forecast(steps=periods)
    return pd.DataFrame({
        'ds': pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=periods),
        'yhat': forecast.predicted_mean,
        'yhat_lower': forecast.conf_int()['lower Close'],
        'yhat_upper': forecast.conf_int()['upper Close']
    })

def holt_winters_forecast(df, periods, trend, seasonal, seasonal_periods):
    model = ExponentialSmoothing(df['Close'], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    results = model.fit()
    forecast = results.forecast(periods)
    resid = df['Close'] - results.fittedvalues
    std_error = np.sqrt(np.sum(resid**2) / (len(df) - 1))
    conf_int = 1.96 * std_error
    return pd.DataFrame({
        'ds': pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=periods),
        'yhat': forecast,
        'yhat_lower': forecast - conf_int,
        'yhat_upper': forecast + conf_int
    })

def sarima_forecast(df, periods, order, seasonal_order):
    model = SARIMAX(df['Close'], order=order, seasonal_order=seasonal_order)
    results = model.fit()
    forecast = results.get_forecast(steps=periods)
    return pd.DataFrame({
        'ds': pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=periods),
        'yhat': forecast.predicted_mean,
        'yhat_lower': forecast.conf_int()['lower Close'],
        'yhat_upper': forecast.conf_int()['upper Close']
    })

def prophet_forecast(df, periods, changepoint_prior_scale, seasonality_prior_scale):
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def random_forest_forecast(df, periods):
    X, y, scaler = prepare_data(df)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    future_X = np.tile(X[-1], (periods, 1))
    
    forecast = model.predict(future_X)
    std_error = np.std(y - model.predict(X))
    conf_int = 1.96 * std_error
    
    return pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast,
        'yhat_lower': forecast - conf_int,
        'yhat_upper': forecast + conf_int
    })

def perform_forecast(df, algorithm, periods=30, **params):
    forecast_functions = {
        'ARIMA': arima_forecast,
        'Holt-Winters': holt_winters_forecast,
        'SARIMA': sarima_forecast,
        'Prophet': prophet_forecast,
        'Random Forest': random_forest_forecast
    }
    
    forecast = forecast_functions[algorithm](df, periods, **params)
    
    historical_forecast = pd.DataFrame({
        'ds': df['Date'],
        'yhat': df['Close'],
        'yhat_lower': df['Close'],
        'yhat_upper': df['Close']
    })
    forecast = pd.concat([historical_forecast, forecast]).reset_index(drop=True)
    
    return forecast

st.set_page_config(layout="wide")
st.title('📈 Stock Forecasting: Multiple Algorithms')

st.sidebar.header('Input Parameters')
stock_market = st.sidebar.selectbox('Select Stock Market', ['US', 'India'])
tickers = us_tickers if stock_market == 'US' else indian_tickers
selected_ticker = st.sidebar.selectbox('Select Stock Ticker', tickers)

if st.sidebar.button('Random Stock'):
    selected_ticker = random.choice(tickers)
    st.sidebar.write(f"Randomly selected stock: {selected_ticker}")

end_date = datetime.now().date()
max_days = 365 * 2
window_size = st.sidebar.slider('Select Window Size (days)', 30, max_days, 90)
start_date = end_date - timedelta(days=window_size)

forecast_horizon = st.sidebar.slider('Forecast Horizon (days)', 1, 365, 30)

forecasting_algorithm = st.sidebar.selectbox('Select Forecasting Algorithm', 
                                             ['ARIMA', 'Holt-Winters', 'SARIMA', 'Prophet', 'Random Forest'])

if forecasting_algorithm == 'ARIMA':
    p = st.sidebar.slider('p (AR order)', 0, 5, 1)
    d = st.sidebar.slider('d (Differencing)', 0, 2, 1)
    q = st.sidebar.slider('q (MA order)', 0, 5, 1)
    params = {'order': (p, d, q)}
elif forecasting_algorithm == 'Holt-Winters':
    trend = st.sidebar.selectbox('Trend', ['add', 'mul', None])
    seasonal = st.sidebar.selectbox('Seasonal', ['add', 'mul', None])
    seasonal_periods = st.sidebar.slider('Seasonal Periods', 1, 30, 7)
    params = {'trend': trend, 'seasonal': seasonal, 'seasonal_periods': seasonal_periods}
elif forecasting_algorithm == 'SARIMA':
    p = st.sidebar.slider('p (AR order)', 0, 5, 1)
    d = st.sidebar.slider('d (Differencing)', 0, 2, 1)
    q = st.sidebar.slider('q (MA order)', 0, 5, 1)
    P = st.sidebar.slider('P (Seasonal AR order)', 0, 2, 1)
    D = st.sidebar.slider('D (Seasonal Differencing)', 0, 1, 1)
    Q = st.sidebar.slider('Q (Seasonal MA order)', 0, 2, 1)
    m = st.sidebar.slider('m (Seasonal Period)', 1, 30, 7)
    params = {'order': (p, d, q), 'seasonal_order': (P, D, Q, m)}
elif forecasting_algorithm == 'Prophet':
    changepoint_prior_scale = st.sidebar.slider('Changepoint Prior Scale', 0.001, 0.5, 0.04)
    seasonality_prior_scale = st.sidebar.slider('Seasonality Prior Scale', 0.01, 10.0, 0.64)
    params = {'changepoint_prior_scale': changepoint_prior_scale, 'seasonality_prior_scale': seasonality_prior_scale}
else:
    params = {}

plot_type = st.sidebar.selectbox('Select Plot Type for Historical Data', ['Line', 'Bar'])

df = get_stock_data(selected_ticker, start_date, end_date)

if df is not None and not df.empty:
    try:
        forecast = perform_forecast(df, forecasting_algorithm, periods=forecast_horizon, **params)
        
        fig = create_forecast_plot(df, forecast, plot_type=plot_type.lower())
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader('Forecast Details')
        forecast_display = forecast.rename(columns={
            'ds': 'Date',
            'yhat': 'Forecast',
            'yhat_lower': 'Lower Bound',
            'yhat_upper': 'Upper Bound'
        })
        st.dataframe(forecast_display.style.format({
            col: '{:.2f}' for col in forecast_display.columns if col != 'Date'
        }), use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred during forecasting: {str(e)}")
        st.error("Please try adjusting the parameters or check your data.")
else:
    st.error(f"No data available for the specified stock and date range.")

st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        margin: 0 auto;
        padding: 0 20px;
    }
    .stSidebar {
        background-color: #f1f3f6;
        padding: 20px;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    .stPlotlyChart {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)