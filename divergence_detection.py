import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import random
import ta

# List of 25 US stock exchange tickers and 25 Indian NS tickers
us_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'NVDA', 'HD', 'DIS', 'BAC', 'ADBE', 'NFLX', 'CRM', 'CMCSA', 'XOM', 'VZ', 'KO', 'INTC']
indian_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'LT.NS', 'AXISBANK.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'BAJAJFINSV.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'TECHM.NS', 'HDFCLIFE.NS']

# Function to fetch stock data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = df['Date'].dt.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Function to calculate technical indicators
def calculate_indicators(df):
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['Stochastic_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['Stochastic_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
    
    return df

# Function to detect anomalies
def detect_anomalies(df, use_rsi, use_macd, use_stoch, use_bb):
    anomalies = []
    
    for i in range(20, len(df)):
        window = df.iloc[i-20:i+1]
        
        if use_rsi:
            # RSI Divergence
            if window['Close'].iloc[-1] < window['Close'].min() and window['RSI'].iloc[-1] > window['RSI'].min():
                anomalies.append(('Bullish', 'RSI Divergence', df.index[i]))
            elif window['Close'].iloc[-1] > window['Close'].max() and window['RSI'].iloc[-1] < window['RSI'].max():
                anomalies.append(('Bearish', 'RSI Divergence', df.index[i]))
        
        if use_macd:
            # MACD Divergence
            if window['Close'].iloc[-1] < window['Close'].min() and window['MACD'].iloc[-1] > window['MACD'].min():
                anomalies.append(('Bullish', 'MACD Divergence', df.index[i]))
            elif window['Close'].iloc[-1] > window['Close'].max() and window['MACD'].iloc[-1] < window['MACD'].max():
                anomalies.append(('Bearish', 'MACD Divergence', df.index[i]))
        
        if use_stoch:
            # Stochastic Divergence
            if window['Close'].iloc[-1] < window['Close'].min() and window['Stochastic_K'].iloc[-1] > window['Stochastic_K'].min():
                anomalies.append(('Bullish', 'Stochastic Divergence', df.index[i]))
            elif window['Close'].iloc[-1] > window['Close'].max() and window['Stochastic_K'].iloc[-1] < window['Stochastic_K'].max():
                anomalies.append(('Bearish', 'Stochastic Divergence', df.index[i]))
        
        if use_bb:
            # Bollinger Band Crossovers
            if df['Close'].iloc[i] > df['BB_High'].iloc[i] and df['Close'].iloc[i-1] <= df['BB_High'].iloc[i-1]:
                anomalies.append(('Bearish', 'BB Upper Crossover', df.index[i]))
            elif df['Close'].iloc[i] < df['BB_Low'].iloc[i] and df['Close'].iloc[i-1] >= df['BB_Low'].iloc[i-1]:
                anomalies.append(('Bullish', 'BB Lower Crossover', df.index[i]))
    
    return anomalies

# Function to create anomaly detection plot
def create_anomaly_plot(df, anomalies, show_sma_20, show_sma_50, show_bb):
    fig = go.Figure()
    
    # Plot price
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')))
    
    # Plot indicators
    if show_sma_20:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
    if show_sma_50:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red')))
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name='BB High', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name='BB Low', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name='BB Mid', line=dict(color='green', dash='dot')))
    
    # Plot anomalies
    for anomaly_type, indicator, date in anomalies:
        color = 'green' if anomaly_type == 'Bullish' else 'red'
        symbol = 'triangle-up' if anomaly_type == 'Bullish' else 'triangle-down'
        fig.add_trace(go.Scatter(x=[date], y=[df.loc[date, 'Close']], mode='markers',
                                 marker=dict(size=10, color=color, symbol=symbol),
                                 name=f'{anomaly_type} {indicator}'))
    
    fig.update_layout(
        title='Stock Price and Anomaly Detection',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# Streamlit app
st.set_page_config(layout="wide")

st.title('📈 Advanced Stock Price Anomaly Detection')

# Sidebar inputs
st.sidebar.header('Input Parameters')

ticker = st.sidebar.text_input('Stock Ticker', value='AAPL')

if st.sidebar.button('Select Random Stock'):
    all_tickers = us_tickers + indian_tickers
    ticker = random.choice(all_tickers)
    st.sidebar.write(f"Randomly selected stock: {ticker}")

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input('Start Date', pd.to_datetime('2023-01-01'))
end_date = col2.date_input('End Date', pd.to_datetime('2023-12-31'))

st.sidebar.header('Indicator Visibility')
show_sma_20 = st.sidebar.checkbox('Show SMA 20', value=True)
show_sma_50 = st.sidebar.checkbox('Show SMA 50', value=True)
show_bb = st.sidebar.checkbox('Show Bollinger Bands', value=True)

st.sidebar.header('Anomaly Detection Methods')
use_rsi = st.sidebar.checkbox('RSI Divergence', value=True)
use_macd = st.sidebar.checkbox('MACD Divergence', value=True)
use_stoch = st.sidebar.checkbox('Stochastic Divergence', value=True)
use_bb = st.sidebar.checkbox('Bollinger Band Crossover', value=True)

# Fetch stock data
df = get_stock_data(ticker, start_date, end_date)

if df is not None and not df.empty:
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Detect anomalies
    anomalies = detect_anomalies(df, use_rsi, use_macd, use_stoch, use_bb)
    
    # Create the plot
    fig = create_anomaly_plot(df, anomalies, show_sma_20, show_sma_50, show_bb)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display anomaly summary
    st.header('Anomaly Detection Summary')
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data Points", len(df))
    col2.metric("Anomalies Detected", len(anomalies))
    col3.metric("Anomaly Percentage", f"{len(anomalies)/len(df)*100:.2f}%")
    
    # Display anomaly details
    if anomalies:
        st.subheader('Detected Anomalies')
        anomaly_df = pd.DataFrame(anomalies, columns=['Type', 'Indicator', 'Date'])
        st.dataframe(anomaly_df, use_container_width=True)
    else:
        st.info("No anomalies detected in the given time frame.")
    
    # Display raw data
    with st.expander("View Raw Data with Indicators"):
        st.dataframe(df, use_container_width=True)
else:
    st.error(f"No data available for {ticker} in the specified date range.")

# Add some styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
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
    </style>
    """, unsafe_allow_html=True)