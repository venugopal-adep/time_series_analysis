import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import plotly.graph_objs as go
import random
from adtk.detector import LevelShiftAD, VolatilityShiftAD, SeasonalAD, PersistAD
from adtk.data import validate_series

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
        df = df[['ds', 'y']]
        
        # Remove null values and compress the data
        original_size = len(df)
        df = df.dropna()
        compressed_size = len(df)
        
        if df.empty:
            return None, 0, 0
        
        return df, original_size, compressed_size
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, 0, 0

# Function to create anomaly detection plot
def create_anomaly_plot(df, anomalies, algorithm):
    fig = go.Figure()
    
    # Plot actual data as blue bars
    fig.add_trace(go.Bar(x=df['ds'], y=df['y'], name='Stock Price', marker_color='blue'))
    
    # Plot anomalies as green bars
    anomaly_dates = df['ds'][anomalies]
    anomaly_values = df['y'][anomalies]
    fig.add_trace(go.Bar(x=anomaly_dates, y=anomaly_values, name='Anomalies', marker_color='green'))
    
    fig.update_layout(
        title=f'Stock Price Anomaly Detection using {algorithm}',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        barmode='overlay'
    )
    return fig

# Function to perform anomaly detection based on selected algorithm
def detect_anomalies(df, algorithm, contamination=0.01):
    X = df['y'].values.reshape(-1, 1)
    
    if algorithm == 'Isolation Forest':
        model = IsolationForest(contamination=contamination, random_state=42)
        anomalies = model.fit_predict(X) == -1
    
    elif algorithm == 'One-Class SVM':
        model = OneClassSVM(nu=contamination)
        anomalies = model.fit_predict(X) == -1
    
    elif algorithm == 'Local Outlier Factor':
        model = LocalOutlierFactor(contamination=contamination)
        anomalies = model.fit_predict(X) == -1
    
    elif algorithm == 'Elliptic Envelope':
        model = EllipticEnvelope(contamination=contamination, random_state=42)
        anomalies = model.fit_predict(X) == -1
    
    elif algorithm == 'Z-Score':
        z_scores = np.abs(stats.zscore(X.ravel()))
        anomalies = z_scores > 3
    
    elif algorithm == 'Seasonal Decomposition':
        result = seasonal_decompose(df['y'], model='additive', period=30)
        residuals = result.resid
        threshold = 2 * np.std(residuals)
        anomalies = np.abs(residuals) > threshold
    
    elif algorithm == 'ADTK Level Shift':
        series = validate_series(df.set_index('ds')['y'])
        level_shift_ad = LevelShiftAD(c=1.0, side='both', window=5)
        anomalies = level_shift_ad.fit_detect(series)
    
    elif algorithm == 'ADTK Volatility Shift':
        series = validate_series(df.set_index('ds')['y'])
        volatility_shift_ad = VolatilityShiftAD(c=1.0, side='both', window=30)
        anomalies = volatility_shift_ad.fit_detect(series)
    
    elif algorithm == 'ADTK Seasonal':
        series = validate_series(df.set_index('ds')['y'])
        seasonal_ad = SeasonalAD(c=1.0, side='both')
        anomalies = seasonal_ad.fit_detect(series)
    
    elif algorithm == 'ADTK Persist':
        series = validate_series(df.set_index('ds')['y'])
        persist_ad = PersistAD(c=3.0, side='positive', window=1)
        anomalies = persist_ad.fit_detect(series)
    
    # Convert anomalies to boolean array and handle NaN values
    if isinstance(anomalies, pd.Series):
        anomalies = anomalies.fillna(False).values
    
    return anomalies

# Streamlit app
st.title('Advanced Stock Price Anomaly Detection')

# Sidebar inputs
ticker = st.sidebar.text_input('Stock Ticker', value='AAPL')

# Add "Select Stock" button
if st.sidebar.button('Select Random Stock'):
    all_tickers = us_tickers + indian_tickers
    ticker = random.choice(all_tickers)
    st.sidebar.write(f"Randomly selected stock: {ticker}")

start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-12-31'))

# Algorithm selection
algorithm_options = ['All Algos', 'Isolation Forest', 'One-Class SVM', 'Local Outlier Factor', 
                     'Elliptic Envelope', 'Z-Score', 'Seasonal Decomposition',
                     'ADTK Level Shift', 'ADTK Volatility Shift', 'ADTK Seasonal', 'ADTK Persist']
algorithm = st.sidebar.selectbox('Anomaly Detection Algorithm', algorithm_options)

# Contamination factor
contamination = st.sidebar.slider('Contamination Factor', min_value=0.01, max_value=0.1, value=0.01, step=0.01)

# Fetch stock data
df, original_size, compressed_size = get_stock_data(ticker, start_date, end_date)

if df is not None and not df.empty:
    # Display data compression information
    st.subheader('Data Compression')
    st.write(f"Original number of samples: {original_size}")
    st.write(f"Compressed number of samples: {compressed_size}")
    st.write(f"Number of samples removed: {original_size - compressed_size}")
    
    if algorithm != 'All Algos':
        # Perform anomaly detection for a single algorithm
        try:
            anomalies = detect_anomalies(df, algorithm, contamination)

            # Create the plot
            fig = create_anomaly_plot(df, anomalies, algorithm)
            st.plotly_chart(fig)

            # Display summary
            st.subheader('Anomaly Detection Summary')
            st.write(f"Total data points: {len(df)}")
            st.write(f"Number of anomalies detected: {sum(anomalies)}")
            st.write(f"Percentage of anomalies: {sum(anomalies)/len(df)*100:.2f}%")

        except Exception as e:
            st.error(f"Error in anomaly detection: {str(e)}")

    else:
        # Perform anomaly detection for all algorithms and create a comparison table
        st.subheader('Algorithm Comparison')
        comparison_data = []

        for algo in algorithm_options[1:]:  # Skip 'All Algos'
            try:
                anomalies = detect_anomalies(df, algo, contamination)
                
                comparison_data.append({
                    'Algorithm': algo,
                    'Anomalies Detected': sum(anomalies),
                    'Percentage': f"{sum(anomalies)/len(df)*100:.2f}%"
                })
            except Exception as e:
                st.warning(f"Error in {algo} anomaly detection: {str(e)}")
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display the table
            st.table(comparison_df)
        else:
            st.warning("No valid anomaly detection results could be generated for any algorithm.")

    # Display raw data
    st.subheader('Raw Data')
    st.write(df)
else:
    st.error(f"No data available for {ticker} in the specified date range.")