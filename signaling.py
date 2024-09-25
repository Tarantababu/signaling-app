import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import csv
import io
import time
import threading

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
        self.positions = []
        self.signals = []

    def calculate_ema(self, data):
        ema = ta.ema(data['Close'], length=self.ema_period)
        return ema

    def calculate_deviation(self, price, ema):
        return (price - ema) / ema if ema != 0 else 0

    def enter_trade(self, direction, price, ema, index):
        self.positions.append({
            'direction': direction,
            'entry_price': price,
            'entry_ema': ema,
            'entry_index': index
        })
        self.signals.append({
            'index': index,
            'signal': 'BUY' if direction == 'long' else 'SELL',
            'price': price,
            'ema': ema
        })

    def check_exit_condition(self, current_price, current_ema, index):
        for position in self.positions[:]:
            if position['direction'] == 'long':
                if current_price <= position['entry_ema'] * (1 - self.stop_loss_percent / 100):
                    self.exit_trade(position, current_price, current_ema, index, 'Stop Loss')
                elif current_price >= position['entry_ema'] * (1 + self.threshold):
                    self.exit_trade(position, current_price, current_ema, index, 'Take Profit')
            else:  # short position
                if current_price >= position['entry_ema'] * (1 + self.stop_loss_percent / 100):
                    self.exit_trade(position, current_price, current_ema, index, 'Stop Loss')
                elif current_price <= position['entry_ema'] * (1 - self.threshold):
                    self.exit_trade(position, current_price, current_ema, index, 'Take Profit')

    def exit_trade(self, position, current_price, current_ema, index, reason):
        self.positions.remove(position)
        self.signals.append({
            'index': index,
            'signal': f"EXIT {position['direction'].upper()}",
            'price': current_price,
            'ema': current_ema,
            'reason': reason
        })

    def generate_signal(self, data):
        ema_series = self.calculate_ema(data)
        signals = []

        for i in range(len(data)):
            price = data['Close'].iloc[i]
            ema = ema_series.iloc[i]
            deviation = self.calculate_deviation(price, ema)

            if not self.positions and abs(deviation) > self.threshold:
                if deviation < 0:
                    self.enter_trade('long', price, ema, i)
                else:
                    self.enter_trade('short', price, ema, i)
            elif self.positions:
                self.check_exit_condition(price, ema, i)

        latest_signal = self.signals[-1] if self.signals else None
        latest_price = data['Close'].iloc[-1]
        latest_ema = ema_series.iloc[-1]
        latest_deviation = self.calculate_deviation(latest_price, latest_ema)

        return latest_signal, latest_price, latest_ema, latest_deviation

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
    signal, price, ema, deviation = generator.generate_signal(data)
    return {
        "ticker": ticker,
        "signal": signal['signal'] if signal else None,
        "price": price,
        "ema": ema,
        "deviation": deviation,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Streamlit app
st.title("Signals")

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
    error_placeholder = st.empty()
    
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
            error_message = f"Error updating signal for {ticker}: {str(e)}"
            error_placeholder.error(error_message)
            
            # Create a new thread to clear the error message after 3 seconds
            threading.Thread(target=clear_error, args=(error_placeholder, 3)).start()
        
        # Update progress
        progress = (i + 1) / total_tickers
        progress_bar.progress(progress)
    
    status_text.text("All signals refreshed!")
    progress_bar.empty()

def clear_error(placeholder, delay):
    time.sleep(delay)
    placeholder.empty()

# Main page
st.header("3:30 PM to 10:00 PM - 2:30 PM to 9:00 PM")

# Single refresh button for all tickers
if st.button("Refresh All Signals"):
    refresh_signals()

# Display Watchlist Summary
st.header("Watchlist Summary")

if st.session_state.tickers:
    watchlist_data = []
    for ticker, ticker_data in st.session_state.tickers.items():
        signal_info = ticker_data["last_signal"] if ticker_data["last_signal"] else {}
        watchlist_data.append({
            "Ticker": ticker,
            "EMA Period": ticker_data["ema_period"],
            "Threshold": f"{ticker_data['threshold']:.2f}",
            "Stop Loss %": f"{ticker_data['stop_loss_percent']:.1f}",
            "Price Threshold": f"{ticker_data['price_threshold']:.3f}",
            "Last Signal": signal_info.get("signal", "N/A"),
            "Last Price": f"{signal_info.get('price', 0):.2f}" if signal_info.get('price') else "N/A",
            "Current EMA": f"{signal_info.get('ema', 0):.2f}" if signal_info.get('ema') else "N/A",
            "Current Deviation": f"{signal_info.get('deviation', 0):.4f}" if signal_info.get('deviation') is not None else "N/A",
            "Timestamp": signal_info.get("timestamp", "N/A"),
            "Actions": ticker
        })
    
    df = pd.DataFrame(watchlist_data)
    
    # Use Streamlit's data editor for inline editing and deletion
    edited_df = st.data_editor(
        df,
        column_config={
            "Actions": st.column_config.Column(
                "Actions",
                width="medium",
                help="Edit or delete this ticker",
                disabled=True,
            )
        },
        hide_index=True,
        num_rows="dynamic"
    )
    
    # Process edits and deletions
    if not df.equals(edited_df):
        error_occurred = False
        for index, row in edited_df.iterrows():
            ticker = row['Ticker']
            try:
                ema_period = int(row['EMA Period'])
                threshold = float(row['Threshold'])
                stop_loss_percent = float(row['Stop Loss %'])
                price_threshold = float(row['Price Threshold'])
                
                if ticker in st.session_state.tickers:
                    st.session_state.tickers[ticker].update({
                        "ema_period": ema_period,
                        "threshold": threshold,
                        "stop_loss_percent": stop_loss_percent,
                        "price_threshold": price_threshold
                    })
                else:
                    st.session_state.tickers[ticker] = {
                        "ema_period": ema_period,
                        "threshold": threshold,
                        "stop_loss_percent": stop_loss_percent,
                        "price_threshold": price_threshold,
                        "last_signal": None
                    }
            except ValueError as e:
                st.error(f"Error processing row for {ticker}: {str(e)}. Please enter valid numeric values.")
                error_occurred = True
        
        # Remove deleted tickers
        tickers_to_remove = set(st.session_state.tickers.keys()) - set(edited_df['Ticker'])
        for ticker in tickers_to_remove:
            del st.session_state.tickers[ticker]
        
        if not error_occurred:
            st.success("Watchlist updated successfully!")
        
        st.experimental_rerun()

else:
    st.write("No tickers added yet. Use the sidebar to add tickers or import a watchlist.")

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

# Export and Import section in the sidebar
st.sidebar.header("Export/Import Watchlist")
if st.sidebar.button("Export Watchlist (CSV)"):
    csv_data = export_watchlist_to_csv()
    st.sidebar.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="watchlist.csv",
        mime="text/csv"
    )

uploaded_file = st.sidebar.file_uploader("Import Watchlist (CSV)", type="csv")
if uploaded_file is not None:
    import_watchlist_from_csv(uploaded_file)

# Run the app
if __name__ == "__main__":
    pass
