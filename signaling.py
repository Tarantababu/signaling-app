import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import csv
import io

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="streamlit")

class SignalGenerator:
    def __init__(self, ticker, ema_period, threshold, stop_loss_percent, price_threshold):
        self.ticker = ticker
        self.ema_period = ema_period
        self.threshold = threshold
        self.stop_loss_percent = stop_loss_percent
        self.price_threshold = price_threshold

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
            price_change = abs(latest_close - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
            if price_change >= self.price_threshold:
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

def generate_signal(ticker, ema_period, threshold, stop_loss_percent, price_threshold):
    generator = SignalGenerator(ticker, ema_period, threshold, stop_loss_percent, price_threshold)
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
st.sidebar.header("Add New Tickers")
new_tickers = st.sidebar.text_input("Ticker Symbols (comma-separated)").upper()
ema_period = st.sidebar.number_input("EMA Period", min_value=1, value=20)
threshold = st.sidebar.number_input("Deviation Threshold", min_value=0.01, value=0.02, format="%.2f")
stop_loss_percent = st.sidebar.number_input("Stop Loss %", min_value=0.1, value=2.0, format="%.1f")
price_threshold = st.sidebar.number_input("Price Threshold", min_value=0.001, value=0.005, format="%.3f")

if st.sidebar.button("Add Tickers"):
    new_ticker_list = [ticker.strip() for ticker in new_tickers.split(',') if ticker.strip()]
    for new_ticker in new_ticker_list:
        if new_ticker:
            st.session_state.tickers[new_ticker] = {
                "ema_period": ema_period,
                "threshold": threshold,
                "stop_loss_percent": stop_loss_percent,
                "price_threshold": price_threshold,
                "last_signal": None
            }
    if new_ticker_list:
        st.success(f"Added {', '.join(new_ticker_list)} to the watchlist.")
    else:
        st.warning("No valid tickers entered.")

# Function to refresh signals
def refresh_signals():
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
                ticker_data["stop_loss_percent"],
                ticker_data["price_threshold"]
            )
            st.session_state.tickers[ticker]["last_signal"] = signal_data
        except Exception as e:
            st.error(f"Error updating signal for {ticker}: {str(e)}")
        
        # Update progress
        progress = (i + 1) / total_tickers
        progress_bar.progress(progress)
    
    status_text.text("All signals refreshed!")
    progress_bar.empty()

# CSV Export Function
def export_watchlist_to_csv():
    data = []
    for ticker, ticker_data in st.session_state.tickers.items():
        data.append({
            "Ticker": ticker,
            "EMA Period": ticker_data["ema_period"],
            "Threshold": ticker_data["threshold"],
            "Stop Loss %": ticker_data["stop_loss_percent"],
            "Price Threshold": ticker_data["price_threshold"]
        })
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

# CSV Import Function
def import_watchlist_from_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        for _, row in df.iterrows():
            ticker = row['Ticker']
            st.session_state.tickers[ticker] = {
                "ema_period": int(row['EMA Period']),
                "threshold": float(row['Threshold']),
                "stop_loss_percent": float(row['Stop Loss %']),
                "price_threshold": float(row['Price Threshold']),
                "last_signal": None
            }
        st.success(f"Imported {len(df)} tickers to the watchlist.")
    except Exception as e:
        st.error(f"Error importing watchlist: {str(e)}")

# Main page
st.header("Trading Signals")

# CSV Export and Import
col1, col2 = st.columns(2)
with col1:
    if st.button("Export Watchlist (CSV)"):
        csv_data = export_watchlist_to_csv()
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="watchlist.csv",
            mime="text/csv"
        )

with col2:
    uploaded_file = st.file_uploader("Import Watchlist (CSV)", type="csv")
    if uploaded_file is not None:
        import_watchlist_from_csv(uploaded_file)

# Single refresh button for all tickers
if st.button("Refresh All Signals"):
    refresh_signals()

# Display active signal notifications
active_signals = [ticker_data["last_signal"] for ticker_data in st.session_state.tickers.values() 
                  if ticker_data["last_signal"] and ticker_data["last_signal"]["signal"]]

if active_signals:
    st.subheader("Active Signals")
    for signal in active_signals:
        st.info(f"🚨 {signal['ticker']}: {signal['signal']} signal at {signal['price']:.2f} ({signal['timestamp']})")

# Display watchlist as a table with edit and delete options
st.header("Watchlist Summary")

if st.session_state.tickers:
    for ticker, ticker_data in list(st.session_state.tickers.items()):
        st.subheader(f"{ticker}")
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"EMA Period: {ticker_data['ema_period']}")
            st.write(f"Threshold: {ticker_data['threshold']:.2f}")
            st.write(f"Stop Loss %: {ticker_data['stop_loss_percent']:.1f}")
            st.write(f"Price Threshold: {ticker_data['price_threshold']:.3f}")
        
        with col2:
            if st.button(f"Edit {ticker}"):
                st.session_state.editing = ticker
        
        with col3:
            if st.button(f"Delete {ticker}"):
                del st.session_state.tickers[ticker]
                st.success(f"Deleted {ticker} from the watchlist.")
                st.experimental_rerun()
        
        if 'editing' in st.session_state and st.session_state.editing == ticker:
            with st.form(f"edit_{ticker}"):
                new_ema_period = st.number_input("New EMA Period", min_value=1, value=ticker_data['ema_period'])
                new_threshold = st.number_input("New Threshold", min_value=0.01, value=ticker_data['threshold'], format="%.2f")
                new_stop_loss = st.number_input("New Stop Loss %", min_value=0.1, value=ticker_data['stop_loss_percent'], format="%.1f")
                new_price_threshold = st.number_input("New Price Threshold", min_value=0.001, value=ticker_data['price_threshold'], format="%.3f")
                
                if st.form_submit_button("Update"):
                    st.session_state.tickers[ticker].update({
                        "ema_period": new_ema_period,
                        "threshold": new_threshold,
                        "stop_loss_percent": new_stop_loss,
                        "price_threshold": new_price_threshold
                    })
                    st.success(f"Updated parameters for {ticker}")
                    del st.session_state.editing
                    st.experimental_rerun()
        
        st.write("---")

    # Display summary table
    summary_data = []
    for ticker, ticker_data in st.session_state.tickers.items():
        signal_info = ticker_data["last_signal"] if ticker_data["last_signal"] else {}
        summary_data.append({
            "Ticker": ticker,
            "Last Signal": signal_info.get("signal", "N/A"),
            "Last Price": f"{signal_info.get('price', 0):.2f}" if signal_info.get('price') else "N/A",
            "Timestamp": signal_info.get("timestamp", "N/A")
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)

else:
    st.write("No tickers added yet. Use the sidebar to add tickers or import a watchlist.")

# Run the app
if __name__ == "__main__":
    pass
