import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from data_prep import simple_moving_average
# Bollinger Bands
'''
A measure of volitility
'''
def bollinger_bands(df, window=20, num_std=2):
    df['BB_MA'] = simple_moving_average(df, window)
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_MA'] + (df['BB_Std'] * num_std)
    df['BB_Lower'] = df['BB_MA'] - (df['BB_Std'] * num_std)
    return df

# Average True Range
'''
price volatility indicator showing the average price variation of assets within a given time period
'''
# def atr(df, window=14):
#     df['H-L'] = df['High'] - df['Low']
#     df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
#     df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
#     df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
#     df['ATR'] = df['TR'].rolling(window=window).mean()
#     return df

# Moving Average Convergence Divergence
'''
trend following indicator
'''
def macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

# Relative Strength Index
'''
momentum oscillator
'''
def rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def mean_reversion_strategy(df, ma_window=20, z_score_window=20, rsi_window=14, 
                            z_score_threshold=1.5, rsi_oversold=40, rsi_overbought=60,
                            bb_window=20, bb_std=2, atr_window=14, macd_fast=12, 
                            macd_slow=26, macd_signal=9, vol_window=20):
    df['MA'] = simple_moving_average(df, ma_window)
    df['Z Score'] = (df['Close'] - df['MA']) / df['Close'].rolling(window=z_score_window).std()
    
    df = bollinger_bands(df, window=bb_window, num_std=bb_std)
    # df = atr(df, window=atr_window)
    df = macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df = rsi(df, window=rsi_window)
    
    df['Volume_SMA'] = df['Volume'].rolling(window=vol_window).mean()
    
    df['Long'] = (
        (df['Z Score'] < -z_score_threshold) & 
        (df['RSI'] < rsi_oversold) &
        ((df['Close'] < df['BB_Lower']) | (df['MACD'] > df['MACD_Signal']))
    )
    
    df['Short'] = (
        (df['Z Score'] > z_score_threshold) & 
        (df['RSI'] > rsi_overbought) &
        ((df['Close'] > df['BB_Upper']) | (df['MACD'] < df['MACD_Signal']))
    )
    
    df['Signal'] = np.where(df['Long'], 1, np.where(df['Short'], -1, 0))
    
    df['Position'] = df['Signal'].diff()
    return df

def backtest(df, initial_capital=10000):
    df['Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Return'] * df['Signal'].shift(1)
    df['Equity'] = (1 + df['Strategy Return'].fillna(0)).cumprod() * initial_capital
    df['Drawdown'] = (df['Equity'] - df['Equity'].cummax()) / df['Equity'].cummax()
    
    total_return = (df['Equity'].iloc[-1] - initial_capital) / initial_capital
    strategy_return_std = df['Strategy Return'].std()
    if strategy_return_std != 0:
        sharpe_ratio = np.sqrt(252) * df['Strategy Return'].mean() / strategy_return_std
    else:
        sharpe_ratio = np.nan
    
    max_drawdown = df['Drawdown'].min()
    
    return df, {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }