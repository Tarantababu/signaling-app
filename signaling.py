import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta

class SignalGenerator:
    def __init__(self, ticker, ema_period, threshold, stop_loss_percent):
        self.ticker = ticker
        self.ema_period = ema_period
        self.threshold = threshold
        self.stop_loss_percent = stop_loss_percent

    def calculate_ema(self, data):
        ema = ta.ema(data['Close'], length=self.ema_period)
        return ema.iloc[-1] if not ema.empty else None

    def calculate_deviation(self, price, ema):
        return (price - ema) / ema if ema is not None else 0

    def generate_signal(self, data):
        if data.empty:
            return None, None

        latest_ema = self.calculate_ema(data)
        latest_close = data['Close'].iloc[-1]
        deviation = self.calculate_deviation(latest_close, latest_ema)

        signal = None
        if abs(deviation) > self.threshold:
            if deviation < 0:
                signal = "BUY"
            else:
                signal = "SELL"

        return signal, latest_close

def fetch_data(ticker, period="1d", interval="1m"):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

def generate_signal(ticker, ema_period, threshold, stop_loss_percent):
    generator = SignalGenerator(ticker, ema_period, threshold, stop_loss_percent)
    data = fetch_data(ticker)
    signal, price = generator.generate_signal(data)
    return {
        "ticker": ticker,
        "signal": signal,
        "price": price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Streamlit app
st.title("Trading Signal App")

# Initialize session state
if 'tickers' not in st.session_state:
    st.session_state.tickers = {}

# Sidebar for adding new tickers
st.sidebar.header("Add New Ticker")
new_ticker = st.sidebar.text_input("Ticker Symbol").upper()
ema_period = st.sidebar.number_input("EMA Period", min_value=1, value=20)
threshold = st.sidebar.number_input("Deviation Threshold", min_value=0.01, value=0.02, format="%.2f")
stop_loss_percent = st.sidebar.number_input("Stop Loss %", min_value=0.1, value=2.0, format="%.1f")

if st.sidebar.button("Add Ticker"):
    if new_ticker:
        st.session_state.tickers[new_ticker] = {
            "ema_period": ema_period,
            "threshold": threshold,
            "stop_loss_percent": stop_loss_percent,
            "last_signal": None
        }
        st.success(f"Added {new_ticker} to the watchlist.")

# Main page
st.header("Active Signals")

# Single refresh button for all tickers
if st.button("Refresh All Signals"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = len(st.session_state.tickers)
    for i, (ticker, ticker_data) in enumerate(st.session_state.tickers.items()):
        status_text.text(f"Refreshing signal for {ticker}...")
        try:
            signal_data = generate_signal(
                ticker, 
                ticker_data["ema_period"], 
                ticker_data["threshold"], 
                ticker_data["stop_loss_percent"]
            )
            st.session_state.tickers[ticker]["last_signal"] = signal_data
        except Exception as e:
            st.error(f"Error updating signal for {ticker}: {str(e)}")
        
        # Update progress
        progress = (i + 1) / total_tickers
        progress_bar.progress(progress)
    
    status_text.text("All signals refreshed!")
    progress_bar.empty()

# Display signals and parameters for each ticker
if st.session_state.tickers:
    for ticker, ticker_data in st.session_state.tickers.items():
        st.subheader(ticker)
        
        # Display parameters
        st.write(f"Parameters:")
        st.write(f"- EMA Period: {ticker_data['ema_period']}")
        st.write(f"- Threshold: {ticker_data['threshold']:.2f}")
        st.write(f"- Stop Loss %: {ticker_data['stop_loss_percent']:.1f}")
        
        # Display signal
        if ticker_data["last_signal"]:
            signal_data = ticker_data["last_signal"]
            if signal_data["signal"]:
                st.write(f"Signal: {signal_data['signal']}")
                st.write(f"Price: {signal_data['price']:.2f}")
                st.write(f"Timestamp: {signal_data['timestamp']}")
            else:
                st.write("No active signal")
        else:
            st.write("Waiting for first signal...")
        
        st.write("---")  # Add a separator between tickers
else:
    st.write("No tickers added yet. Use the sidebar to add tickers.")

# Run the app
if __name__ == "__main__":
    pass
