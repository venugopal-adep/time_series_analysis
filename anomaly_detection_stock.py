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
    
    # Plot actual data as a line
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Stock Price', mode='lines', line=dict(color='#3366CC')))
    
    # Plot anomalies as red dots
    anomaly_dates = df['ds'][anomalies]
    anomaly_values = df['y'][anomalies]
    fig.add_trace(go.Scatter(x=anomaly_dates, y=anomaly_values, name='Anomalies', mode='markers', marker=dict(color='red', size=8)))
    
    fig.update_layout(
        title=f'Stock Price Anomaly Detection using {algorithm}',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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
    
    if isinstance(anomalies, pd.Series):
        anomalies = anomalies.fillna(False).values
    
    return anomalies

# Set page config
st.set_page_config(page_title="Stock Anomaly Detector", layout="wide")

# Custom CSS for improved aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #3366CC;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app
st.title('🚀 Advanced Stock Price Anomaly Detection')

# Sidebar inputs
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input('Stock Ticker', value='AAPL')

    if st.button('Select Random Stock'):
        all_tickers = us_tickers + indian_tickers
        ticker = random.choice(all_tickers)
        st.write(f"Randomly selected stock: {ticker}")

    start_date = st.date_input('Start Date', pd.to_datetime('2023-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2023-12-31'))

    algorithm_options = ['All Algos', 'Isolation Forest', 'One-Class SVM', 'Local Outlier Factor', 
                         'Elliptic Envelope', 'Z-Score', 'Seasonal Decomposition',
                         'ADTK Level Shift', 'ADTK Volatility Shift', 'ADTK Seasonal', 'ADTK Persist']
    algorithm = st.selectbox('Anomaly Detection Algorithm', algorithm_options)

    contamination = st.slider('Contamination Factor', min_value=0.01, max_value=0.1, value=0.01, step=0.01)

# Fetch stock data
df, original_size, compressed_size = get_stock_data(ticker, start_date, end_date)

if df is not None and not df.empty:
    # Display data compression information
    st.subheader('📊 Data Compression')
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Samples", original_size)
    col2.metric("Compressed Samples", compressed_size)
    col3.metric("Samples Removed", original_size - compressed_size)
    
    if algorithm != 'All Algos':
        # Perform anomaly detection for a single algorithm
        try:
            anomalies = detect_anomalies(df, algorithm, contamination)

            # Create the plot
            fig = create_anomaly_plot(df, anomalies, algorithm)
            st.plotly_chart(fig, use_container_width=True)

            # Display summary
            st.subheader('🔍 Anomaly Detection Summary')
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Data Points", len(df))
            col2.metric("Anomalies Detected", sum(anomalies))
            col3.metric("Percentage of Anomalies", f"{sum(anomalies)/len(df)*100:.2f}%")

        except Exception as e:
            st.error(f"Error in anomaly detection: {str(e)}")

    else:
        # Perform anomaly detection for all algorithms and create a comparison table
        st.subheader('🔬 Algorithm Comparison')
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
    st.subheader('📈 Raw Data')
    st.dataframe(df)
else:
    st.error(f"No data available for {ticker} in the specified date range.")
