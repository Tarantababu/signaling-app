import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import io
import time
import threading

# Function to calculate EMA manually
def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

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
        return calculate_ema(data['Close'], self.ema_period)

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
                elif current_price >= position['entry_ema']:
                    self.exit_trade(position, current_price, current_ema, index, 'Take Profit')
            else:  # short position
                if current_price >= position['entry_ema'] * (1 + self.stop_loss_percent / 100):
                    self.exit_trade(position, current_price, current_ema, index, 'Stop Loss')
                elif current_price <= position['entry_ema']:
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
        if data.empty:
            return None, None, None, None
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

def show_temporary_message(message, message_type="info", duration=3):
    placeholder = st.empty()
    if message_type == "success":
        placeholder.success(message)
    elif message_type == "warning":
        placeholder.warning(message)
    elif message_type == "error":
        placeholder.error(message)
    else:
        placeholder.info(message)
    
    threading.Timer(duration, placeholder.empty).start()

def fetch_data(ticker, period="1d", interval="1m"):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            show_temporary_message(f"No data available for {ticker}", "warning")
        return data
    except Exception as e:
        show_temporary_message(f"Error fetching data for {ticker}: {str(e)}", "error")
        return pd.DataFrame()

def generate_signal(ticker, ema_period, threshold, stop_loss_percent, price_threshold):
    generator = SignalGenerator(ticker, ema_period, threshold, stop_loss_percent, price_threshold)
    data = fetch_data(ticker)
    if data.empty:
        return {
            "ticker": ticker,
            "signal": None,
            "price": None,
            "ema": None,
            "deviation": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    signal, price, ema, deviation = generator.generate_signal(data)
    
    result = {
        "ticker": ticker,
        "signal": signal['signal'] if signal else None,
        "price": price,
        "ema": ema,
        "deviation": deviation,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Update active signal
    if result["signal"] in ["BUY", "SELL"]:
        st.session_state.tickers[ticker]["active_signal"] = result
    elif result["signal"] and "EXIT" in result["signal"]:
        if st.session_state.tickers[ticker]["active_signal"]:
            st.session_state.tickers[ticker]["active_signal"] = None
    
    return result

def create_chart(ticker, ema_period, threshold, stop_loss_percent, price_threshold):
    data = fetch_data(ticker, period="5d", interval="5m")
    if data.empty:
        show_temporary_message(f"No data available to create chart for {ticker}", "warning")
        return None
    
    # Calculate EMA
    ema = calculate_ema(data['Close'], ema_period)
    
    # Create price-based candles
    price_based_data = []
    current_candle = data.iloc[0].copy()
    last_price = current_candle['Close']

    for i in range(1, len(data)):
        row = data.iloc[i]
        price_change = abs(row['Close'] - last_price) / last_price
        if price_change >= price_threshold:
            current_candle['EMA'] = ema.iloc[i-1]
            price_based_data.append(current_candle)
            current_candle = row.copy()
            last_price = row['Close']
        else:
            current_candle['High'] = max(current_candle['High'], row['High'])
            current_candle['Low'] = min(current_candle['Low'], row['Low'])
            current_candle['Close'] = row['Close']
            current_candle['Volume'] += row['Volume']

    current_candle['EMA'] = ema.iloc[-1]
    price_based_data.append(current_candle)
    price_based_df = pd.DataFrame(price_based_data)

    # Calculate deviation
    price_based_df['Deviation'] = (price_based_df['Close'] - price_based_df['EMA']) / price_based_df['EMA']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Price-Based Candles and EMA', 'Deviation'),
                        row_heights=[0.7, 0.3])
    
    # Price-based candles and EMA
    fig.add_trace(go.Candlestick(x=price_based_df.index, open=price_based_df['Open'], high=price_based_df['High'],
                                 low=price_based_df['Low'], close=price_based_df['Close'], name='Price'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=price_based_df.index, y=price_based_df['EMA'], name=f'EMA-{ema_period}', line=dict(color='orange')),
                  row=1, col=1)
    
    # Deviation
    fig.add_trace(go.Scatter(x=price_based_df.index, y=price_based_df['Deviation'], name='Deviation', line=dict(color='purple')),
                  row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="red", row=2, col=1)
    
    # Add vertical lines for time separation
    time_intervals = pd.date_range(start=price_based_df.index.min(), end=price_based_df.index.max(), freq='D')
    for time in time_intervals:
        fig.add_vline(x=time, line_width=1, line_dash="dash", line_color="rgba(100,100,100,0.2)", row="all")

    # Signals
    for i in range(1, len(price_based_df)):
        if abs(price_based_df['Deviation'].iloc[i]) > threshold:
            if price_based_df['Deviation'].iloc[i] < 0:
                fig.add_trace(go.Scatter(x=[price_based_df.index[i]], y=[price_based_df['Low'].iloc[i]], mode='markers',
                                         marker=dict(symbol='triangle-up', size=10, color='green'),
                                         name='Buy Signal'),
                              row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=[price_based_df.index[i]], y=[price_based_df['High'].iloc[i]], mode='markers',
                                         marker=dict(symbol='triangle-down', size=10, color='red'),
                                         name='Sell Signal'),
                              row=1, col=1)

        # Exit signals when price hits EMA
        if (price_based_df['Close'].iloc[i-1] < price_based_df['EMA'].iloc[i-1] and 
            price_based_df['Close'].iloc[i] >= price_based_df['EMA'].iloc[i]):
            fig.add_trace(go.Scatter(x=[price_based_df.index[i]], y=[price_based_df['High'].iloc[i]], mode='markers',
                                     marker=dict(symbol='square', size=8, color='blue'),
                                     name='Long Exit'),
                          row=1, col=1)
        elif (price_based_df['Close'].iloc[i-1] > price_based_df['EMA'].iloc[i-1] and 
              price_based_df['Close'].iloc[i] <= price_based_df['EMA'].iloc[i]):
            fig.add_trace(go.Scatter(x=[price_based_df.index[i]], y=[price_based_df['Low'].iloc[i]], mode='markers',
                                     marker=dict(symbol='square', size=8, color='orange'),
                                     name='Short Exit'),
                          row=1, col=1)

    fig.update_layout(height=800, title_text=f"{ticker} Price-Based Analysis")
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def display_active_signals():
    st.sidebar.header("Active Signals")
    active_signals = []
    for ticker, ticker_data in st.session_state.tickers.items():
        signal_info = ticker_data.get("active_signal")
        if signal_info and signal_info["signal"] in ["BUY", "SELL"]:
            sl_price = signal_info["price"] * (1 - ticker_data["stop_loss_percent"]/100) if signal_info["signal"] == "BUY" else signal_info["price"] * (1 + ticker_data["stop_loss_percent"]/100)
            active_signals.append({
                "Ticker": ticker,
                "Signal": signal_info['signal'],
                "Entry": f"{signal_info['price']:.2f}",
                "Stop Loss": f"{sl_price:.2f}",
                "Timestamp": signal_info['timestamp']
            })

    if active_signals:
        df = pd.DataFrame(active_signals)
        st.sidebar.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.sidebar.info("No active signals at the moment.")

def refresh_signals():
    progress_bar = st.empty()
    status_text = st.empty()
    
    total_tickers = len(st.session_state.tickers)
    tickers_to_remove = []
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
            show_temporary_message(error_message, "error")
            tickers_to_remove.append(ticker)
        
        progress = (i + 1) / total_tickers
        progress_bar.progress(progress)
    
    for ticker in tickers_to_remove:
        del st.session_state.tickers[ticker]
    
    if tickers_to_remove:
        show_temporary_message(f"Removed {len(tickers_to_remove)} ticker(s) due to data fetching errors.", "warning")
    
    status_text.empty()
    progress_bar.empty()

    # Refresh the display of active signals
    display_active_signals()

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
                "last_signal": None,
                "active_signal": None
            }
        show_temporary_message(f"Imported {len(df)} tickers to the watchlist.", "success")
    except Exception as e:
        show_temporary_message(f"Error importing watchlist: {str(e)}", "error")

def main():
    st.title("Signals")

    # Initialize session state
    if 'tickers' not in st.session_state:
        st.session_state.tickers = {}
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None

    # Sidebar for adding new tickers
    st.sidebar.header("Add New Tickers")
    new_tickers = st.sidebar.text_input("Ticker Symbols (comma-separated)").upper()
    ema_period = st.sidebar.number_input("EMA Period", min_value=1, value=20)
    threshold = st.sidebar.number_input("Deviation Threshold", min_value=0.000001, value=0.02, format="%.6f")
    stop_loss_percent = st.sidebar.number_input("Stop Loss %", min_value=0.1, value=2.0, format="%.1f")
    price_threshold = st.sidebar.number_input("Price Threshold", min_value=0.000001, value=0.005, format="%.6f")

    if st.sidebar.button("Add Tickers"):
        new_ticker_list = [ticker.strip() for ticker in new_tickers.split(',') if ticker.strip()]
        for new_ticker in new_ticker_list:
            if new_ticker:
                st.session_state.tickers[new_ticker] = {
                    "ema_period": ema_period,
                    "threshold": threshold,
                    "stop_loss_percent": stop_loss_percent,
                    "price_threshold": price_threshold,
                    "last_signal": None,
                    "active_signal": None
                }
        if new_ticker_list:
            show_temporary_message(f"Added {', '.join(new_ticker_list)} to the watchlist.", "success")
            refresh_signals()  # Refresh signals for the new tickers
        else:
            show_temporary_message("No valid tickers entered.", "warning")

    # Display Active Signals
    display_active_signals()

    # Export and Import section in the sidebar
    if st.sidebar.button("Export Watchlist (CSV)"):
        csv_data = export_watchlist_to_csv()
        st.sidebar.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="watchlist.csv",
            mime="text/csv"
        )
        show_temporary_message("Watchlist exported successfully.", "success")

    uploaded_file = st.sidebar.file_uploader("Import Watchlist (CSV)", type="csv")
    if uploaded_file is not None:
        import_watchlist_from_csv(uploaded_file)
        refresh_signals()  # Refresh signals for the imported tickers

    # Main page
    st.header("...")

    # Single refresh button for all tickers
    if st.button("Refresh All Signals"):
        refresh_signals()
        show_temporary_message("All signals refreshed!", "success")

    # Display Watchlist Summary
    st.header("Watchlist Summary")

    if st.session_state.tickers:
        watchlist_data = []
        for ticker, ticker_data in st.session_state.tickers.items():
            signal_info = ticker_data["last_signal"] if ticker_data["last_signal"] else {}
            watchlist_data.append({
                "Ticker": ticker,
                "EMA Period": ticker_data["ema_period"],
                "Threshold": f"{ticker_data['threshold']:.6f}",
                "Stop Loss %": f"{ticker_data['stop_loss_percent']:.1f}",
                "Price Threshold": f"{ticker_data['price_threshold']:.6f}",
                "Last Signal": signal_info.get("signal", "N/A"),
                "Last Price": f"{signal_info.get('price', 0):.2f}" if signal_info.get('price') else "N/A",
                "Current EMA": f"{signal_info.get('ema', 0):.2f}" if signal_info.get('ema') else "N/A",
                "Current Deviation": f"{signal_info.get('deviation', 0):.6f}" if signal_info.get('deviation') is not None else "N/A",
                "Timestamp": signal_info.get("timestamp", "N/A"),
            })
        
        df = pd.DataFrame(watchlist_data)
        
        # Use Streamlit's data editor for inline editing and deletion
        edited_df = st.data_editor(
            df,
            hide_index=True,
            num_rows="dynamic",
            key="watchlist_table"
        )
        
        # Create a selectbox for choosing a ticker to display
        selected_ticker = st.selectbox("Select a ticker to display chart", options=list(st.session_state.tickers.keys()))
        
        # Display chart for selected ticker
        if selected_ticker:
            st.subheader(f"Chart for {selected_ticker}")
            ticker_data = st.session_state.tickers[selected_ticker]
            chart = create_chart(
                selected_ticker,
                ticker_data["ema_period"],
                ticker_data["threshold"],
                ticker_data["stop_loss_percent"],
                ticker_data["price_threshold"]
            )
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        
        # Process edits and deletions
        if not df.equals(edited_df):
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
                            "last_signal": None,
                            "active_signal": None
                        }
                except ValueError as e:
                    show_temporary_message(f"Error processing row for {ticker}: {str(e)}. This ticker will be ignored.", "error")
                    if ticker in st.session_state.tickers:
                        del st.session_state.tickers[ticker]
            
            # Remove deleted tickers
            tickers_to_remove = set(st.session_state.tickers.keys()) - set(edited_df['Ticker'])
            for ticker in tickers_to_remove:
                del st.session_state.tickers[ticker]
            
            show_temporary_message("Watchlist updated successfully!", "success")
            refresh_signals()  # Refresh signals after updating the watchlist
            st.experimental_rerun()

    else:
        st.write("No tickers added yet. Use the sidebar to add tickers or import a watchlist.")

if __name__ == "__main__":
    main()
