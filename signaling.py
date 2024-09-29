import streamlit as st, yfinance as yf, pandas as pd, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json, threading

def calculate_ema(data, period): return data.ewm(span=period, adjust=False).mean()

class SignalGenerator:
    def __init__(self, ticker, ema_period, threshold, stop_loss_percent, price_threshold):
        self.ticker, self.ema_period, self.threshold, self.stop_loss_percent, self.price_threshold = ticker, ema_period, threshold, stop_loss_percent, price_threshold
        self.positions, self.signals = [], []
    def calculate_ema(self, data): return calculate_ema(data['Close'], self.ema_period)
    def calculate_deviation(self, price, ema): return (price - ema) / ema if ema != 0 else 0
    def enter_trade(self, direction, price, ema, index):
        self.positions.append({'direction': direction, 'entry_price': price, 'entry_ema': ema, 'entry_index': index})
        self.signals.append({'index': index, 'signal': 'BUY' if direction == 'long' else 'SELL', 'price': price, 'ema': ema})
    def check_exit_condition(self, current_price, current_ema, index):
        for position in self.positions[:]:
            if position['direction'] == 'long':
                if current_price <= position['entry_ema'] * (1 - self.stop_loss_percent / 100) or current_price >= position['entry_ema']:
                    self.exit_trade(position, current_price, current_ema, index, 'Stop Loss' if current_price <= position['entry_ema'] * (1 - self.stop_loss_percent / 100) else 'Take Profit')
            else:
                if current_price >= position['entry_ema'] * (1 + self.stop_loss_percent / 100) or current_price <= position['entry_ema']:
                    self.exit_trade(position, current_price, current_ema, index, 'Stop Loss' if current_price >= position['entry_ema'] * (1 + self.stop_loss_percent / 100) else 'Take Profit')
    def exit_trade(self, position, current_price, current_ema, index, reason):
        self.positions.remove(position)
        self.signals.append({'index': index, 'signal': f"EXIT {position['direction'].upper()}", 'price': current_price, 'ema': current_ema, 'reason': reason})
    def generate_signal(self, data):
        if data.empty: return None, None, None, None
        ema_series = self.calculate_ema(data)
        for i in range(len(data)):
            price, ema = data['Close'].iloc[i], ema_series.iloc[i]
            deviation = self.calculate_deviation(price, ema)
            if not self.positions and abs(deviation) > self.threshold:
                self.enter_trade('long' if deviation < 0 else 'short', price, ema, i)
            elif self.positions: self.check_exit_condition(price, ema, i)
        latest_signal = self.signals[-1] if self.signals else None
        return latest_signal, data['Close'].iloc[-1], ema_series.iloc[-1], self.calculate_deviation(data['Close'].iloc[-1], ema_series.iloc[-1])

class Profile:
    def __init__(self, name):
        self.name, self.positions, self.trade_history, self.watchlist = name, [], [], {}
    def add_position(self, ticker, entry_price, quantity, direction):
        self.positions.append({'ticker': ticker, 'entry_price': entry_price, 'quantity': quantity, 'direction': direction, 'entry_time': datetime.now().isoformat()})
    def close_position(self, index, exit_price, reason):
        position = self.positions.pop(index)
        position.update({'exit_price': exit_price, 'exit_time': datetime.now().isoformat(), 'reason': reason, 'pl': self.calculate_pl(position, exit_price)})
        self.trade_history.append(position)
        return position['pl']
    def delete_position(self, index): return self.positions.pop(index)
    def calculate_pl(self, position, current_price):
        return (current_price - position['entry_price']) * position['quantity'] if position['direction'] == 'long' else (position['entry_price'] - current_price) * position['quantity']
    def add_to_watchlist(self, ticker, ema_period, threshold, stop_loss_percent, price_threshold):
        self.watchlist[ticker] = {"ema_period": ema_period, "threshold": threshold, "stop_loss_percent": stop_loss_percent, "price_threshold": price_threshold, "last_signal": None, "active_signal": None}
    def remove_from_watchlist(self, ticker):
        if ticker in self.watchlist: del self.watchlist[ticker]
    def to_dict(self): return {'name': self.name, 'positions': self.positions, 'trade_history': self.trade_history, 'watchlist': self.watchlist}
    @classmethod
    def from_dict(cls, data):
        profile = cls(data['name'])
        profile.positions, profile.trade_history, profile.watchlist = data.get('positions', []), data.get('trade_history', []), data.get('watchlist', {})
        return profile

def save_profile(profile):
    st.session_state.profiles[profile.name] = profile.to_dict()
    with open('profiles.json', 'w') as f: json.dump(st.session_state.profiles, f)

def load_profiles():
    try:
        with open('profiles.json', 'r') as f: return json.load(f)
    except FileNotFoundError: return {}

def delete_profile(profile_name):
    if profile_name in st.session_state.profiles:
        del st.session_state.profiles[profile_name]
        with open('profiles.json', 'w') as f: json.dump(st.session_state.profiles, f)
        return True
    return False

def rename_profile(old_name, new_name):
    if old_name in st.session_state.profiles and new_name not in st.session_state.profiles:
        st.session_state.profiles[new_name] = st.session_state.profiles.pop(old_name)
        st.session_state.profiles[new_name]['name'] = new_name
        with open('profiles.json', 'w') as f: json.dump(st.session_state.profiles, f)
        if st.session_state.current_profile and st.session_state.current_profile.name == old_name:
            st.session_state.current_profile.name = new_name
        return True
    return False

def show_temporary_message(message, message_type="info", duration=3):
    placeholder = st.empty()
    getattr(placeholder, message_type)(message)
    threading.Timer(duration, placeholder.empty).start()

def fetch_data(ticker, period="1d", interval="1m"):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty: show_temporary_message(f"No data available for {ticker}", "warning")
        return data
    except Exception as e:
        show_temporary_message(f"Error fetching data for {ticker}: {str(e)}", "error")
        return pd.DataFrame()

def generate_signal(ticker, ema_period, threshold, stop_loss_percent, price_threshold):
    generator = SignalGenerator(ticker, ema_period, threshold, stop_loss_percent, price_threshold)
    data = fetch_data(ticker)
    if data.empty: return {"ticker": ticker, "signal": None, "price": None, "ema": None, "deviation": None, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    signal, price, ema, deviation = generator.generate_signal(data)
    return {"ticker": ticker, "signal": signal['signal'] if signal else None, "price": price, "ema": ema, "deviation": deviation, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

def create_chart(ticker, ema_period, threshold, stop_loss_percent, price_threshold):
    data = fetch_data(ticker, period="5d", interval="5m")
    if data.empty:
        show_temporary_message(f"No data available to create chart for {ticker}", "warning")
        return None
    ema = calculate_ema(data['Close'], ema_period)
    price_based_data, current_candle, last_price = [], data.iloc[0].copy(), data.iloc[0]['Close']
    for i in range(1, len(data)):
        row = data.iloc[i]
        if abs(row['Close'] - last_price) / last_price >= price_threshold:
            current_candle['EMA'] = ema.iloc[i-1]
            price_based_data.append(current_candle)
            current_candle, last_price = row.copy(), row['Close']
        else:
            current_candle['High'], current_candle['Low'], current_candle['Close'], current_candle['Volume'] = max(current_candle['High'], row['High']), min(current_candle['Low'], row['Low']), row['Close'], current_candle['Volume'] + row['Volume']
    current_candle['EMA'] = ema.iloc[-1]
    price_based_data.append(current_candle)
    price_based_df = pd.DataFrame(price_based_data)
    price_based_df['Deviation'] = (price_based_df['Close'] - price_based_df['EMA']) / price_based_df['EMA']
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=(f'{ticker} Price-Based Candles and EMA', 'Deviation'), row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=price_based_df.index, open=price_based_df['Open'], high=price_based_df['High'], low=price_based_df['Low'], close=price_based_df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_based_df.index, y=price_based_df['EMA'], name=f'EMA-{ema_period}', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_based_df.index, y=price_based_df['Deviation'], name='Deviation', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="red", row=2, col=1)
    for time in pd.date_range(start=price_based_df.index.min(), end=price_based_df.index.max(), freq='D'):
        fig.add_vline(x=time, line_width=1, line_dash="dash", line_color="rgba(100,100,100,0.2)", row="all")
    for i in range(1, len(price_based_df)):
        if abs(price_based_df['Deviation'].iloc[i]) > threshold:
            fig.add_trace(go.Scatter(x=[price_based_df.index[i]], y=[price_based_df['Low' if price_based_df['Deviation'].iloc[i] < 0 else 'High'].iloc[i]], mode='markers', marker=dict(symbol='triangle-up' if price_based_df['Deviation'].iloc[i] < 0 else 'triangle-down', size=10, color='green' if price_based_df['Deviation'].iloc[i] < 0 else 'red'), name='Buy Signal' if price_based_df['Deviation'].iloc[i] < 0 else 'Sell Signal'), row=1, col=1)
        if (price_based_df['Close'].iloc[i-1] < price_based_df['EMA'].iloc[i-1] and price_based_df['Close'].iloc[i] >= price_based_df['EMA'].iloc[i]) or (price_based_df['Close'].iloc[i-1] > price_based_df['EMA'].iloc[i-1] and price_based_df['Close'].iloc[i] <= price_based_df['EMA'].iloc[i]):
            fig.add_trace(go.Scatter(x=[price_based_df.index[i]], y=[price_based_df['High' if price_based_df['Close'].iloc[i-1] < price_based_df['EMA'].iloc[i-1] else 'Low'].iloc[i]], mode='markers', marker=dict(symbol='square', size=8, color='blue' if price_based_df['Close'].iloc[i-1] < price_based_df['EMA'].iloc[i-1] else 'orange'), name='Long Exit' if price_based_df['Close'].iloc[i-1] < price_based_df['EMA'].iloc[i-1] else 'Short Exit'), row=1, col=1)
    fig.update_layout(height=800, title_text=f"{ticker} Price-Based Analysis")
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def refresh_signals():
    progress_bar, status_text = st.empty(), st.empty()
    total_tickers = len(st.session_state.tickers)
    tickers_to_remove, active_signals, exit_signals_for_open_positions = [], [], []
    for i, (ticker, ticker_data) in enumerate(st.session_state.tickers.items()):
        status_text.text(f"Refreshing signal for {ticker}...")
        try:
            signal_data = generate_signal(ticker, ticker_data["ema_period"], ticker_data["threshold"], ticker_data["stop_loss_percent"], ticker_data["price_threshold"])
            st.session_state.tickers[ticker]["last_signal"] = signal_data
            if signal_data["signal"] in ["BUY", "SELL"]:
                active_signals.append({"Ticker": ticker, "Signal": signal_data['signal'], "Price": f"${signal_data['price']:.2f}", "EMA": f"${signal_data['ema']:.2f}", "Deviation": f"{signal_data['deviation']:.4f}", "Timestamp": signal_data['timestamp']})
            if signal_data["signal"] in ["EXIT LONG", "EXIT SHORT"]:
                for position in st.session_state.current_profile.positions:
                    if position['ticker'] == ticker:
                        exit_signals_for_open_positions.append({"Ticker": ticker, "Signal": f"EXIT {'Long' if position['direction'] == 'long' else 'Short'}", "Price": f"${signal_data['price']:.2f}", "EMA": f"${signal_data['ema']:.2f}", "Deviation": f"{signal_data['deviation']:.4f}", "Timestamp": signal_data['timestamp']})
                        break
        except Exception as e:
            show_temporary_message(f"Error updating signal for {ticker}: {str(e)}", "error")
            tickers_to_remove.append(ticker)
        progress_bar.progress((i + 1) / total_tickers)
    for ticker in tickers_to_remove: del st.session_state.tickers[ticker]
    if tickers_to_remove: show_temporary_message(f"Removed {len(tickers_to_remove)} ticker(s) due to data fetching errors.", "warning")
    status_text.empty(); progress_bar.empty()
    st.session_state.active_signals = active_signals
    st.session_state.exit_signals_for_open_positions = exit_signals_for_open_positions

def display_active_signals():
    st.subheader("Active Signals")
    if hasattr(st.session_state, 'active_signals') and st.session_state.active_signals:
        st.dataframe(pd.DataFrame(st.session_state.active_signals), hide_index=True, use_container_width=True)
    else: st.info("No active signals at the moment.")
    if hasattr(st.session_state, 'exit_signals_for_open_positions') and st.session_state.exit_signals_for_open_positions:
        st.subheader("Exit Signals for Open Positions", help="These are exit signals for currently open positions.")
        st.dataframe(pd.DataFrame(st.session_state.exit_signals_for_open_positions), hide_index=True, use_container_width=True)
        st.markdown('<p style="color:green;">These exit signals are for your open positions.</p>', unsafe_allow_html=True)

def fetch_current_price(ticker):
    try: return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching current price for {ticker}: {str(e)}")
        return 0

def display_profile(profile):
    st.header(f"Profile: {profile.name}")
    st.subheader("Open Positions")
    if profile.positions:
        positions_data = [{"Index": i, "Ticker": p['ticker'], "Direction": p['direction'], "Quantity": p['quantity'], "Entry Price": f"${p['entry_price']:.2f}", "Current Price": f"${fetch_current_price(p['ticker']):.2f}", "P/L": f"${profile.calculate_pl(p, fetch_current_price(p['ticker'])):.2f}", "Entry Time": p['entry_time']} for i, p in enumerate(profile.positions)]
        positions_df = pd.DataFrame(positions_data)
        positions_df['Entry Time'] = pd.to_datetime(positions_df['Entry Time'])
        st.dataframe(positions_df.sort_values('Entry Time', ascending=False), hide_index=True, use_container_width=True)
        st.subheader("Position Actions")
        col1, col2, col3 = st.columns(3)
        with col1: action = st.selectbox("Select Action", ["Close Position", "Delete Position"])
        with col2: position_index = st.selectbox("Select Position", options=positions_df['Index'].tolist())
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
    else: st.info("No open positions.")
    st.subheader("Trade History")
    if profile.trade_history:
        history_df = pd.DataFrame(profile.trade_history)
        history_df['entry_time'], history_df['exit_time'] = pd.to_datetime(history_df['entry_time']), pd.to_datetime(history_df['exit_time'])
        st.dataframe(history_df.sort_values('exit_time', ascending=False))
    else: st.info("No trade history.")
    st.subheader("Add New Position")
    col1, col2, col3, col4 = st.columns(4)
    with col1: ticker = st.text_input("Ticker")
    with col2: entry_price = st.number_input("Entry Price", min_value=0.01, step=0.01)
    with col3: quantity = st.number_input("Quantity", min_value=1, step=1)
    with col4: direction = st.selectbox("Direction", ["long", "short"])
    if st.button("Add Position"):
        profile.add_position(ticker, entry_price, quantity, direction)
        save_profile(profile)
        st.success(f"Position added for {ticker}")
        st.experimental_rerun()

def check_exit_signals(profile, signal_generator):
    positions_to_close = []
    for i, position in enumerate(profile.positions):
        ticker, direction, entry_price = position['ticker'], position['direction'], position['entry_price']
        signal_data = signal_generator(ticker, st.session_state.tickers[ticker]["ema_period"], st.session_state.tickers[ticker]["threshold"], st.session_state.tickers[ticker]["stop_loss_percent"], st.session_state.tickers[ticker]["price_threshold"])
        current_price, current_ema = signal_data['price'], signal_data['ema']
        if direction == 'long':
            if current_price <= current_ema: positions_to_close.append((i, current_price, "Exit Long"))
            elif current_price <= entry_price * (1 - st.session_state.tickers[ticker]["stop_loss_percent"] / 100): positions_to_close.append((i, current_price, "Stop Loss"))
        elif direction == 'short':
            if current_price >= current_ema: positions_to_close.append((i, current_price, "Exit Short"))
            elif current_price >= entry_price * (1 + st.session_state.tickers[ticker]["stop_loss_percent"] / 100): positions_to_close.append((i, current_price, "Stop Loss"))
    for index, price, reason in reversed(positions_to_close):
        pl = profile.close_position(index, price, reason)
        st.warning(f"{reason} triggered for {profile.positions[index]['ticker']} at {price:.2f}. P/L: ${pl:.2f}")
    if positions_to_close: save_profile(profile)

def profile_management():
    if 'profiles' not in st.session_state: st.session_state.profiles = load_profiles()
    if 'current_profile' not in st.session_state: st.session_state.current_profile = None
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
            else: st.error("Please enter a unique profile name.")
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
                else: st.sidebar.error("Failed to rename profile. Name might already exist.")
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

def main():
    st.title("Signals")
    if 'tickers' not in st.session_state: st.session_state.tickers = {}
    if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = None
    if 'active_signals' not in st.session_state: st.session_state.active_signals = []
    if 'exit_signals_for_open_positions' not in st.session_state: st.session_state.exit_signals_for_open_positions = []
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
                    st.session_state.current_profile.add_to_watchlist(new_ticker, ema_period, threshold, stop_loss_percent, price_threshold)
            if new_ticker_list:
                save_profile(st.session_state.current_profile)
                st.session_state.tickers = st.session_state.current_profile.watchlist
                show_temporary_message(f"Added {', '.join(new_ticker_list)} to the watchlist.", "success")
                st.experimental_rerun()
            else: show_temporary_message("No valid tickers entered.", "warning")
        if st.sidebar.button("Export Watchlist (CSV)"):
            csv_data = pd.DataFrame([{"Ticker": t, "EMA Period": d["ema_period"], "Threshold": d["threshold"], "Stop Loss %": d["stop_loss_percent"], "Price Threshold": d["price_threshold"]} for t, d in st.session_state.tickers.items()]).to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(label="Download CSV", data=csv_data, file_name="watchlist.csv", mime="text/csv")
            show_temporary_message("Watchlist exported successfully.", "success")
        uploaded_file = st.sidebar.file_uploader("Import Watchlist (CSV)", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                for _, row in df.iterrows():
                    st.session_state.current_profile.add_to_watchlist(row['Ticker'], int(row['EMA Period']), float(row['Threshold']), float(row['Stop Loss %']), float(row['Price Threshold']))
                save_profile(st.session_state.current_profile)
                st.session_state.tickers = st.session_state.current_profile.watchlist
                show_temporary_message(f"Imported {len(df)} tickers to the watchlist.", "success")
                st.experimental_rerun()
            except Exception as e: show_temporary_message(f"Error importing watchlist: {str(e)}", "error")
        st.header("Watchlist and Signals")
        if st.button("Refresh All Signals"):
            refresh_signals()
            check_exit_signals(st.session_state.current_profile, generate_signal)
            show_temporary_message("All signals refreshed!", "success")
            st.experimental_rerun()
        display_active_signals()
        st.header("Watchlist Summary")
        if st.session_state.current_profile.watchlist:
            watchlist_data = [{"Ticker": ticker, "EMA Period": ticker_data["ema_period"], "Threshold": f"{ticker_data['threshold']:.6f}", "Stop Loss %": f"{ticker_data['stop_loss_percent']:.1f}", "Price Threshold": f"{ticker_data['price_threshold']:.6f}", "Last Signal": signal_info.get("signal", "N/A"), "Last Price": f"{signal_info.get('price', 0):.2f}" if signal_info.get('price') else "N/A", "Current EMA": f"{signal_info.get('ema', 0):.2f}" if signal_info.get('ema') else "N/A", "Current Deviation": f"{signal_info.get('deviation', 0):.6f}" if signal_info.get('deviation') is not None else "N/A", "Timestamp": signal_info.get("timestamp", "N/A")} for ticker, ticker_data in st.session_state.current_profile.watchlist.items() if (signal_info := ticker_data["last_signal"] if ticker_data["last_signal"] else {})]
            df = pd.DataFrame(watchlist_data)
            edited_df = st.data_editor(df, hide_index=True, num_rows="dynamic", key="watchlist_table")
            selected_ticker = st.selectbox("Select a ticker to display chart", options=list(st.session_state.current_profile.watchlist.keys()))
        if selected_ticker:
                st.subheader(f"Chart for {selected_ticker}")
                ticker_data = st.session_state.current_profile.watchlist[selected_ticker]
                chart = create_chart(selected_ticker, ticker_data["ema_period"], ticker_data["threshold"], ticker_data["stop_loss_percent"], ticker_data["price_threshold"])
                if chart: st.plotly_chart(chart, use_container_width=True)
            
            if not df.equals(edited_df):
                for index, row in edited_df.iterrows():
                    ticker = row['Ticker']
                    try:
                        st.session_state.current_profile.add_to_watchlist(ticker, int(row['EMA Period']), float(row['Threshold']), float(row['Stop Loss %']), float(row['Price Threshold']))
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
