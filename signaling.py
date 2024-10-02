import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import io
import time as tm
import threading
import json
import pytz
import requests

TELEGRAM_TOKEN = "7148511647:AAFlMohYiqPF2GQFtri2qW4H0WU2-j174TQ"
TELEGRAM_CHAT_ID = "5611879467"

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def fetch_alpha_vantage_data(ticker, interval="1min"):
    api_key = st.session_state.get('alpha_vantage_api_key', '')
    if not api_key:
        st.error("Alpha Vantage API key is not set. Please enter it in the sidebar.")
        return pd.DataFrame()

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&outputsize=full&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    
    if f"Time Series ({interval})" not in data:
        st.error(f"Error fetching data for {ticker}: {data.get('Note', 'Unknown error')}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data[f"Time Series ({interval})"]).T
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df.sort_index()

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
            else:
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

class Profile:
    def __init__(self, name):
        self.name = name
        self.positions = []
        self.trade_history = []
        self.watchlist = {}

    def add_position(self, ticker, entry_price, quantity, direction):
        position = {
            'ticker': ticker,
            'entry_price': entry_price,
            'quantity': quantity,
            'direction': direction,
            'entry_time': datetime.now().isoformat()
        }
        self.positions.append(position)

    def close_position(self, index, exit_price, reason):
        position = self.positions.pop(index)
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now().isoformat()
        position['reason'] = reason
        pl = self.calculate_pl(position, exit_price)
        position['pl'] = pl
        self.trade_history.append(position)
        return pl

    def delete_position(self, index):
        return self.positions.pop(index)

    def calculate_pl(self, position, current_price):
        if position['direction'] == 'long':
            return (current_price - position['entry_price']) * position['quantity']
        else:
            return (position['entry_price'] - current_price) * position['quantity']

    def add_to_watchlist(self, ticker, ema_period, threshold, stop_loss_percent, price_threshold):
        self.watchlist[ticker] = {
            "ema_period": ema_period,
            "threshold": threshold,
            "stop_loss_percent": stop_loss_percent,
            "price_threshold": price_threshold,
            "last_signal": None,
            "active_signal": None
        }

    def remove_from_watchlist(self, ticker):
        if ticker in self.watchlist:
            del self.watchlist[ticker]

    def to_dict(self):
        return {
            'name': self.name,
            'positions': self.positions,
            'trade_history': self.trade_history,
            'watchlist': self.watchlist
        }

    @classmethod
    def from_dict(cls, data):
        profile = cls(data['name'])
        profile.positions = data.get('positions', [])
        profile.trade_history = data.get('trade_history', [])
        profile.watchlist = data.get('watchlist', {})
        return profile

def save_profile(profile):
    st.session_state.profiles[profile.name] = profile.to_dict()
    try:
        with open('profiles.json', 'w') as f:
            json.dump(st.session_state.profiles, f, indent=4)
    except Exception as e:
        st.error(f"Error saving profiles: {str(e)}")

def load_profiles():
    try:
        with open('profiles.json', 'r') as f:
            content = f.read()
            if not content:
                return {}
            return json.loads(content)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        st.error("Error: profiles.json contains invalid data. Starting with an empty profile list.")
        return {}

def delete_profile(profile_name):
    if profile_name in st.session_state.profiles:
        del st.session_state.profiles[profile_name]
        with open('profiles.json', 'w') as f:
            json.dump(st.session_state.profiles, f, indent=4)
        return True
    return False

def rename_profile(old_name, new_name):
    if old_name in st.session_state.profiles and new_name not in st.session_state.profiles:
        st.session_state.profiles[new_name] = st.session_state.profiles.pop(old_name)
        st.session_state.profiles[new_name]['name'] = new_name
        with open('profiles.json', 'w') as f:
            json.dump(st.session_state.profiles, f, indent=4)
        if st.session_state.current_profile and st.session_state.current_profile.name == old_name:
            st.session_state.current_profile.name = new_name
        return True
    return False

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

def generate_signal(ticker, ema_period, threshold, stop_loss_percent, price_threshold):
    generator = SignalGenerator(ticker, ema_period, threshold, stop_loss_percent, price_threshold)
    data = fetch_alpha_vantage_data(ticker)
    if data.empty:
        return {
            "ticker": ticker,
            "signal": None,
            "price": None,
            "ema": None,
            "deviation": None,
            "sl_price": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    signal, price, ema, deviation = generator.generate_signal(data)
    
    sl_price = None
    if signal and signal['signal'] in ['BUY', 'SELL']:
        if signal['signal'] == 'BUY':
            sl_price = price * (1 - stop_loss_percent / 100)
        else:  # SELL signal
            sl_price = price * (1 + stop_loss_percent / 100)
    
    result = {
        "ticker": ticker,
        "signal": signal['signal'] if signal else None,
        "price": price,
        "ema": ema,
        "deviation": deviation,
        "sl_price": sl_price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result

def create_chart(ticker, ema_period, threshold, stop_loss_percent, price_threshold):
    data = fetch_alpha_vantage_data(ticker)
    if data.empty:
        show_temporary_message(f"No data available to create chart for {ticker}", "warning")
        return None
    
    ema = calculate_ema(data['Close'], ema_period)
    
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

    price_based_df['Deviation'] = (price_based_df['Close'] - price_based_df['EMA']) / price_based_df['EMA']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Price-Based Candles and EMA', 'Deviation'),
                        row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=price_based_df.index, open=price_based_df['Open'], high=price_based_df['High'],
                                 low=price_based_df['Low'], close=price_based_df['Close'], name='Price'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=price_based_df.index, y=price_based_df['EMA'], name=f'EMA-{ema_period}', line=dict(color='orange')),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=price_based_df.index, y=price_based_df['Deviation'], name='Deviation', line=dict(color='purple')),
                  row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="red", row=2, col=1)
    
    time_intervals = pd.date_range(start=price_based_df.index.min(), end=price_based_df.index.max(), freq='D')
    for time in time_intervals:
        fig.add_vline(x=time, line_width=1, line_dash="dash", line_color="rgba(100,100,100,0.2)", row="all")

    buy_signals = []
    sell_signals = []

    for i in range(1, len(price_based_df)):
        if abs(price_based_df['Deviation'].iloc[i]) > threshold:
            if price_based_df['Deviation'].iloc[i] < 0:
                buy_signals.append(i)
                fig.add_trace(go.Scatter(x=[price_based_df.index[i]], y=[price_based_df['Low'].iloc[i]], mode='markers',
                                         marker=dict(symbol='triangle-up', size=10, color='green'),
                                         name='Buy Signal'),
                              row=1, col=1)
            else:
                sell_signals.append(i)
                fig.add_trace(go.Scatter(x=[price_based_df.index[i]], y=[price_based_df['High'].iloc[i]], mode='markers',
                                        marker=dict(symbol='triangle-down', size=10, color='red'),
                                         name='Sell Signal'),
                              row=1, col=1)

    for i in range(1, len(price_based_df)):
        if i in buy_signals:
            for j in range(i+1, len(price_based_df)):
                if price_based_df['Close'].iloc[j] >= price_based_df['EMA'].iloc[j]:
                    fig.add_trace(go.Scatter(x=[price_based_df.index[j]], y=[price_based_df['High'].iloc[j]], mode='markers',
                                             marker=dict(symbol='square', size=8, color='blue'),
                                             name='Long Exit'),
                                  row=1, col=1)
                    break
        elif i in sell_signals:
            for j in range(i+1, len(price_based_df)):
                if price_based_df['Close'].iloc[j] <= price_based_df['EMA'].iloc[j]:
                    fig.add_trace(go.Scatter(x=[price_based_df.index[j]], y=[price_based_df['Low'].iloc[j]], mode='markers',
                                             marker=dict(symbol='square', size=8, color='orange'),
                                             name='Short Exit'),
                                  row=1, col=1)
                    break

    fig.update_layout(height=800, title_text=f"{ticker} Price-Based Analysis")
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def fetch_current_price(ticker):
    try:
        data = fetch_alpha_vantage_data(ticker)
        return data['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching current price for {ticker}: {str(e)}")
        return 0

def get_market_close_time():
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now.time() > time(16, 0) or now.weekday() >= 5:
        days_ahead = 1 if now.weekday() < 4 else (7 - now.weekday())
        market_close += timedelta(days=days_ahead)
    
    if now.time() < time(9, 30):
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_close

def send_new_signals_to_telegram(new_signals):
    if not new_signals:
        return "No new signals to send"

    message = "New Active Signals:\n"
    for signal in new_signals:
        message += f"{signal['Ticker']}: {signal['Signal']} at {signal['Price']} (SL: {signal['SL Price']})\n"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        if res.status_code == 200:
            return "sent"
        else:
            print(f"Telegram API responded with status code {res.status_code}")
            return "failed"
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return "failed"

def refresh_signals():
    progress_bar = st.empty()
    status_text = st.empty()
    
    total_tickers = len(st.session_state.tickers)
    tickers_to_remove = []
    active_signals = []
    new_active_signals = []
    exit_signals_for_open_positions = []
    
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
            
            if signal_data["signal"] in ["BUY", "SELL"]:
                new_signal = {
                    "Ticker": ticker,
                    "Signal": signal_data['signal'],
                    "Price": f"${signal_data['price']:.2f}",
                    "SL Price": f"${signal_data['sl_price']:.2f}" if signal_data['sl_price'] else "N/A",
                    "EMA": f"${signal_data['ema']:.2f}",
                    "Deviation": f"{signal_data['deviation']:.4f}",
                    "Timestamp": signal_data['timestamp']
                }
                active_signals.append(new_signal)
                
                # Check if this is a new signal
                last_signal = ticker_data.get("last_signal", {})
                if last_signal.get("signal") != signal_data["signal"] or last_signal.get("price") != signal_data["price"]:
                    new_active_signals.append(new_signal)
            
            if signal_data["signal"] in ["EXIT LONG", "EXIT SHORT"]:
                for position in st.session_state.current_profile.positions:
                    if position['ticker'] == ticker:
                        exit_type = "Long" if position['direction'] == 'long' else "Short"
                        exit_signals_for_open_positions.append({
                            "Ticker": ticker,
                            "Signal": f"EXIT {exit_type}",
                            "Price": f"${signal_data['price']:.2f}",
                            "EMA": f"${signal_data['ema']:.2f}",
                            "Deviation": f"{signal_data['deviation']:.4f}",
                            "Timestamp": signal_data['timestamp']
                        })
                        break
            
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

    st.session_state.active_signals = active_signals
    st.session_state.exit_signals_for_open_positions = exit_signals_for_open_positions

    # Send new signals to Telegram
    if new_active_signals:
        send_new_signals_to_telegram(new_active_signals)

def auto_refresh():
    if st.session_state.get('auto_refresh', False):
        refresh_signals()
        st.experimental_rerun()

def display_market_close_timer():
    market_close = get_market_close_time()
    now = datetime.now(pytz.timezone('America/New_York'))
    time_remaining = market_close - now
    
    hours, remainder = divmod(time_remaining.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    st.sidebar.markdown("### Time to Market Close")
    st.sidebar.markdown(f"{time_remaining.days} days, {hours:02d}:{minutes:02d}:{seconds:02d}")

    if time_remaining.days == 0 and hours == 0 and minutes < 30:
        st.sidebar.warning("⚠️ Less than 30 minutes until market close!")

def display_profile(profile):
    st.header(f"Profile: {profile.name}")
    
    st.subheader("Open Positions")
    if profile.positions:
        positions_data = []
        total_pl = 0
        for i, position in enumerate(profile.positions):
            current_price = fetch_current_price(position['ticker'])
            pl = profile.calculate_pl(position, current_price)
            total_pl += pl
            positions_data.append({
                "Index": i,
                "Ticker": position['ticker'],
                "Direction": position['direction'],
                "Quantity": position['quantity'],
                "Entry Price": f"${position['entry_price']:.2f}",
                "Current Price": f"${current_price:.2f}",
                "P/L": f"${pl:.2f}",
                "Entry Time": position['entry_time']
            })
        positions_df = pd.DataFrame(positions_data)
        positions_df['Entry Time'] = pd.to_datetime(positions_df['Entry Time'])
        positions_df = positions_df.sort_values('Entry Time', ascending=False)
        
        st.dataframe(positions_df, hide_index=True, use_container_width=True)
        
        st.markdown(f"**Total P/L for Open Positions: ${total_pl:.2f}**")
        
        st.subheader("Position Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            action = st.selectbox("Select Action", ["Close Position", "Delete Position"])
        with col2:
            position_index = st.selectbox("Select Position", options=positions_df['Index'].tolist())
        with col3:
            if action == "Close Position":
                close_price = st.number_input("Close Price", min_value=0.01, step=0.01, value=fetch_current_price(profile.positions[position_index]['ticker']))
                if st.button("Execute"):
                    pl = profile.close_position(position_index, close_price, "Manual Close")
                    save_profile(profile)
                    st.success(f"Position closed with P/L: ${pl:.2f}")
                    st.experimental_rerun()
            else:
                if st.button("Execute"):
                    deleted_position = profile.delete_position(position_index)
                    save_profile(profile)
                    st.success(f"Position for {deleted_position['ticker']} deleted.")
                    st.experimental_rerun()
    else:
        st.info("No open positions.")

    st.subheader("Trade History")
    if profile.trade_history:
        history_df = pd.DataFrame(profile.trade_history)
        history_df['entry_time'] = pd.to_datetime(history_df['entry_time'])
        history_df['exit_time'] = pd.to_datetime(history_df['exit_time'])
        history_df = history_df.sort_values('exit_time', ascending=False)
        st.dataframe(history_df)
    else:
        st.info("No trade history.")

    st.subheader("Add New Position")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ticker = st.text_input("Ticker")
    with col2:
        entry_price = st.number_input("Entry Price", min_value=0.01, step=0.01)
    with col3:
        quantity = st.number_input("Quantity", min_value=1, step=1)
    with col4:
        direction = st.selectbox("Direction", ["long", "short"])
    
    if st.button("Add Position"):
        profile.add_position(ticker, entry_price, quantity, direction)
        save_profile(profile)
        st.success(f"Position added for {ticker}")
        st.experimental_rerun()

def check_exit_signals(profile, signal_generator):
    exit_signals = []
    for position in profile.positions:
        ticker = position['ticker']
        direction = position['direction']
        entry_price = position['entry_price']
        
        signal_data = signal_generator(
            ticker, 
            st.session_state.tickers[ticker]["ema_period"], 
            st.session_state.tickers[ticker]["threshold"], 
            st.session_state.tickers[ticker]["stop_loss_percent"],
            st.session_state.tickers[ticker]["price_threshold"]
        )
        
        current_price = signal_data['price']
        current_ema = signal_data['ema']
        
        if direction == 'long':
            if current_price <= current_ema:
                exit_signals.append((ticker, current_price, "Exit Long"))
            elif current_price <= entry_price * (1 - st.session_state.tickers[ticker]["stop_loss_percent"] / 100):
                exit_signals.append((ticker, current_price, "Stop Loss (Long)"))
        elif direction == 'short':
            if current_price >= current_ema:
                exit_signals.append((ticker, current_price, "Exit Short"))
            elif current_price >= entry_price * (1 + st.session_state.tickers[ticker]["stop_loss_percent"] / 100):
                exit_signals.append((ticker, current_price, "Stop Loss (Short)"))

    return exit_signals

def profile_management():
    if 'profiles' not in st.session_state:
        st.session_state.profiles = load_profiles()
    
    if 'current_profile' not in st.session_state:
        st.session_state.current_profile = None

    st.sidebar.header("Profile Management")
    profile_names = list(st.session_state.profiles.keys())
    profile_names.insert(0, "Create New Profile")
    
    selected_profile = st.sidebar.selectbox("Select Profile", profile_names)
    
    if selected_profile == "Create New Profile":
        new_profile_name = st.sidebar.text_input("Enter new profile name")
        if st.sidebar.button("Create Profile"):
            if new_profile_name and new_profile_name not in st.session_state.profiles:
                st.session_state.current_profile = Profile(new_profile_name)
                save_profile(st.session_state.current_profile)
                st.success(f"Profile '{new_profile_name}' created successfully!")
                st.experimental_rerun()
            else:
                st.error("Please enter a unique profile name.")
    elif selected_profile:
        try:
            st.session_state.current_profile = Profile.from_dict(st.session_state.profiles[selected_profile])
            st.sidebar.success(f"Profile '{selected_profile}' loaded.")
            st.session_state.tickers = st.session_state.current_profile.watchlist
        except Exception as e:
            st.sidebar.error(f"Error loading profile: {str(e)}")
            st.session_state.current_profile = None
            st.session_state.tickers = {}

    if st.session_state.current_profile:
        st.sidebar.subheader("Edit Profile")
        new_name = st.sidebar.text_input("New profile name", value=st.session_state.current_profile.name)
        if st.sidebar.button("Rename Profile"):
            if new_name != st.session_state.current_profile.name:
                if rename_profile(st.session_state.current_profile.name, new_name):
                    st.sidebar.success(f"Profile renamed to '{new_name}'")
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Failed to rename profile. Name might already exist.")

        if st.sidebar.button("Delete Profile"):
            profile_name = st.session_state.current_profile.name
            st.session_state.current_profile = None
            st.session_state.tickers = {}
            if delete_profile(profile_name):
                st.sidebar.success(f"Profile '{profile_name}' deleted")
                st.experimental_rerun()
            else:
                st.sidebar.error("Failed to delete profile")
                st.session_state.current_profile = Profile.from_dict(st.session_state.profiles[profile_name])

        display_profile(st.session_state.current_profile)

def display_active_signals():
    st.subheader("Active Signals")
    if hasattr(st.session_state, 'active_signals') and st.session_state.active_signals:
        df = pd.DataFrame(st.session_state.active_signals)
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No active signals at the moment.")
    
    if hasattr(st.session_state, 'exit_signals_for_open_positions') and st.session_state.exit_signals_for_open_positions:
        st.subheader("Exit Signals for Open Positions", help="These are exit signals for currently open positions. Action required.")
        df_exit = pd.DataFrame(st.session_state.exit_signals_for_open_positions)
        st.dataframe(df_exit, hide_index=True, use_container_width=True)
        st.markdown('<p style="color:red;">Warning: These are exit signals for your open positions. Please review and take appropriate action.</p>', unsafe_allow_html=True)

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

def import_watchlist_from_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        for _, row in df.iterrows():
            ticker = row['Ticker']
            st.session_state.current_profile.add_to_watchlist(
                ticker,
                int(row['EMA Period']),
                float(row['Threshold']),
                float(row['Stop Loss %']),
                float(row['Price Threshold'])
            )
        save_profile(st.session_state.current_profile)
        st.session_state.tickers = st.session_state.current_profile.watchlist
        show_temporary_message(f"Imported {len(df)} tickers to the watchlist.", "success")
    except Exception as e:
        show_temporary_message(f"Error importing watchlist: {str(e)}", "error")

def main():
    st.title("Signals")

    if 'tickers' not in st.session_state:
        st.session_state.tickers = {}
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    if 'active_signals' not in st.session_state:
        st.session_state.active_signals = []
    if 'exit_signals_for_open_positions' not in st.session_state:
        st.session_state.exit_signals_for_open_positions = []
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5

    # Add UI for Alpha Vantage API key input
    st.sidebar.header("API Configuration")
    api_key = st.sidebar.text_input("Enter Alpha Vantage API Key", type="password")
    if api_key:
        st.session_state.alpha_vantage_api_key = api_key

    # Add UI for auto-refresh configuration
    st.sidebar.header("Auto Refresh Configuration")
    st.session_state.auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=st.session_state.auto_refresh)
    st.session_state.refresh_interval = st.sidebar.number_input("Refresh Interval (minutes)", min_value=1, value=st.session_state.refresh_interval)

    display_market_close_timer()
    profile_management()

    if st.session_state.current_profile:
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
                    st.session_state.current_profile.add_to_watchlist(
                        new_ticker,
                        ema_period,
                        threshold,
                        stop_loss_percent,
                        price_threshold
                    )
            if new_ticker_list:
                save_profile(st.session_state.current_profile)
                st.session_state.tickers = st.session_state.current_profile.watchlist
                show_temporary_message(f"Added {', '.join(new_ticker_list)} to the watchlist.", "success")
                st.experimental_rerun()
            else:
                show_temporary_message("No valid tickers entered.", "warning")

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
            st.experimental_rerun()

        st.header("Watchlist and Signals")

        if st.button("Refresh All Signals") or (st.session_state.auto_refresh and st.empty()):
            refresh_signals()
            exit_signals = check_exit_signals(st.session_state.current_profile, generate_signal)
            st.session_state.exit_signals_for_open_positions = [
                {
                    "Ticker": ticker,
                    "Signal": signal,
                    "Price": f"${price:.2f}",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                } for ticker, price, signal in exit_signals
            ]
            show_temporary_message("All signals refreshed!", "success")
            
        if st.session_state.auto_refresh:
            st.write(f"Auto-refreshing every {st.session_state.refresh_interval} minutes")
            auto_refresh()
            st.empty()

        display_active_signals()

        st.header("Watchlist Summary")

        if st.session_state.current_profile.watchlist:
            watchlist_data = []
            for ticker, ticker_data in st.session_state.current_profile.watchlist.items():
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
            
            edited_df = st.data_editor(
                df,
                hide_index=True,
                num_rows="dynamic",
                key="watchlist_table"
            )
            
            selected_ticker = st.selectbox("Select a ticker to display chart", options=list(st.session_state.current_profile.watchlist.keys()))
            
            if selected_ticker:
                st.subheader(f"Chart for {selected_ticker}")
                ticker_data = st.session_state.current_profile.watchlist[selected_ticker]
                chart = create_chart(
                    selected_ticker,
                    ticker_data["ema_period"],
                    ticker_data["threshold"],
                    ticker_data["stop_loss_percent"],
                    ticker_data["price_threshold"]
                )
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            if not df.equals(edited_df):
                for index, row in edited_df.iterrows():
                    ticker = row['Ticker']
                    try:
                        ema_period = int(row['EMA Period'])
                        threshold = float(row['Threshold'])
                        stop_loss_percent = float(row['Stop Loss %'])
                        price_threshold = float(row['Price Threshold'])
                        
                        st.session_state.current_profile.add_to_watchlist(
                            ticker,
                            ema_period,
                            threshold,
                            stop_loss_percent,
                            price_threshold
                        )
                    except ValueError as e:
                        show_temporary_message(f"Error processing row for {ticker}: {str(e)}. This ticker will be ignored.", "error")
                        st.session_state.current_profile.remove_from_watchlist(ticker)
                
                tickers_to_remove = set(st.session_state.current_profile.watchlist.keys()) - set(edited_df['Ticker'])
                for ticker in tickers_to_remove:
                    st.session_state.current_profile.remove_from_watchlist(ticker)
                
                save_profile(st.session_state.current_profile)
                st.session_state.tickers = st.session_state.current_profile.watchlist
                show_temporary_message("Watchlist updated successfully!", "success")
                st.experimental_rerun()

        else:
            st.write("No tickers added yet. Use the sidebar to add tickers or import a watchlist.")
    else:
        st.write("Please select or create a profile to start.")

if __name__ == "__main__":
    main()
