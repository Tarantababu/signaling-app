import streamlit as st
import yfinance as yf
import pandas as pd
try:
    import pandas_ta as ta
except ImportError:
    st.error("Error importing pandas_ta. Please make sure all requirements are installed.")
    st.stop()
from datetime import datetime, timedelta
import time

class SignalGenerator:
    def __init__(self, ticker, ema_period, threshold, stop_loss_percent):
        self.ticker = ticker
        self.ema_period = ema_period
        self.threshold = threshold
        self.stop_loss_percent = stop_loss_percent
        self.current_position = None
        self.entry_price = None
        self.stop_loss = None

    def calculate_ema(self, data):
        return ta.ema(data['Close'], length=self.ema_period)

    def calculate_deviation(self, price, ema):
        return (price - ema) / ema

    def generate_signal(self, data):
        ema = self.calculate_ema(data)
        latest_close = data['Close'].iloc[-1]
        latest_ema = ema.iloc[-1]
        deviation = self.calculate_deviation(latest_close, latest_ema)

        signal = None
        if self.current_position is None:
            if abs(deviation) > self.threshold:
                if deviation < 0:
                    signal = "BUY"
                    self.current_position = "LONG"
                    self.entry_price = latest_close
                    self.stop_loss = latest_close * (1 - self.stop_loss_percent / 100)
                else:
                    signal = "SELL"
                    self.current_position = "SHORT"
                    self.entry_price = latest_close
                    self.stop_loss = latest_close * (1 + self.stop_loss_percent / 100)
        else:
            if (self.current_position == "LONG" and latest_close >= latest_ema) or \
               (self.current_position == "SHORT" and latest_close <= latest_ema):
                signal = "CLOSE"
                self.current_position = None
                self.entry_price = None
                self.stop_loss = None
            elif (self.current_position == "LONG" and latest_close <= self.stop_loss) or \
                 (self.current_position == "SHORT" and latest_close >= self.stop_loss):
                signal = "STOP LOSS"
                self.current_position = None
                self.entry_price = None
                self.stop_loss = None

        return signal, latest_close, self.current_position, self.entry_price, self.stop_loss

def fetch_data(ticker, period="1d", interval="1m"):
    return yf.download(ticker, period=period, interval=interval)

def run_signal_generator(ticker, ema_period, threshold, stop_loss_percent):
    generator = SignalGenerator(ticker, ema_period, threshold, stop_loss_percent)
    while True:
        data = fetch_data(ticker)
        signal, price, position, entry_price, stop_loss = generator.generate_signal(data)
        if signal:
            yield {
                "ticker": ticker,
                "signal": signal,
                "price": price,
                "position": position,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        time.sleep(60)  # Wait for 1 minute before checking again

# Streamlit app
st.title("Trading Signal App")

# Sidebar for adding new tickers
st.sidebar.header("Add New Ticker")
new_ticker = st.sidebar.text_input("Ticker Symbol").upper()
ema_period = st.sidebar.number_input("EMA Period", min_value=1, value=20)
threshold = st.sidebar.number_input("Deviation Threshold", min_value=0.01, value=0.02, format="%.2f")
stop_loss_percent = st.sidebar.number_input("Stop Loss %", min_value=0.1, value=2.0, format="%.1f")

if st.sidebar.button("Add Ticker"):
    if new_ticker:
        if 'tickers' not in st.session_state:
            st.session_state.tickers = {}
        st.session_state.tickers[new_ticker] = {
            "ema_period": ema_period,
            "threshold": threshold,
            "stop_loss_percent": stop_loss_percent,
            "generator": run_signal_generator(new_ticker, ema_period, threshold, stop_loss_percent)
        }
        st.success(f"Added {new_ticker} to the watchlist.")

# Main page
st.header("Active Signals")

if 'tickers' in st.session_state:
    for ticker, ticker_data in st.session_state.tickers.items():
        st.subheader(ticker)
        try:
            signal_data = next(ticker_data["generator"])
            if signal_data["signal"]:
                st.write(f"Signal: {signal_data['signal']}")
                st.write(f"Price: {signal_data['price']:.2f}")
                if signal_data["position"]:
                    st.write(f"Position: {signal_data['position']}")
                    st.write(f"Entry Price: {signal_data['entry_price']:.2f}")
                    st.write(f"Stop Loss: {signal_data['stop_loss']:.2f}")
                st.write(f"Timestamp: {signal_data['timestamp']}")
            else:
                st.write("No active signal")
        except StopIteration:
            st.error(f"Error fetching data for {ticker}")

    if st.button("Refresh Signals"):
        st.experimental_rerun()
else:
    st.write("No tickers added yet. Use the sidebar to add tickers.")

# Run the app
if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
