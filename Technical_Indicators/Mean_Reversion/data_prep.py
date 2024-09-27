import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def fetch_data(ticker, start_date, end_date, interval='1d'):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval=interval)
    return df

def time_series_features(df):
    df2 = df.copy()
    df2['day_of_week'] = df2.index.dayofweek
    df2['month'] = df2.index.month
    df2['year'] = df2.index.year
    df2['day_of_month'] = df2.index.day
    df2['day_of_year'] = df2.index.dayofyear
    # df['week_of_year'] = df.index.weekofyear
    df2['quarter'] = df2.index.quarter
    df2['is_month_start'] = df2.index.is_month_start
    df2['is_month_end'] = df2.index.is_month_end
    return df2

def add_lag_features(df, column='Close', lags=[1, 2, 3, 5, 10]):
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df

def shift_features(df, shift_columns, shift_by=1):
    for col in shift_columns:
        df[f'{col}_shifted'] = df[col].shift(shift_by)
    return df

def simple_moving_average(df, window):
    return df['Close'].rolling(window=window).mean()

def windowed_df_to_date(df):
    dates = df['Date'].values
    features = df.drop(['Date', 'Close'], axis=1).values
    X = features.reshape((len(dates), features.shape[1], 1))
    y = df['Close'].values
    return X.astype(np.float32), y.astype(np.float32), dates

def train_test_split(df, X, y, test_size=0.2):
    X, y, dates = windowed_df_to_date(df)

    q_80 = int(len(df) * 0.8)
    q_90 = int(len(df) * 0.9)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    dates_train = dates[:q_80]
    dates_val = dates[q_80:q_90]
    dates_test = dates[q_90:]

    X_train = X_scaled[:q_80]
    X_val = X_scaled[q_80:q_90]
    X_test = X_scaled[q_90:]

    y_train = y_scaled[:q_80]
    y_val = y_scaled[q_80:q_90]
    y_test = y_scaled[q_90:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test

def apply_jcandle(df):
    df.loc[df['Close'] > df['Open'], 'j_candle'] = 0
    df.loc[df['Close'] < df['Open'], 'j_candle'] = 1
    df.loc[df['j_candle'] == 0, 'high_wick_length'] = df['High'] - df['Close']
    df.loc[df['j_candle'] == 1, 'high_wick_length'] = df['High'] - df['Open']
    df.loc[df['j_candle'] == 0, 'low_wick_length'] = df['Close'] - df['Low']
    df.loc[df['j_candle'] == 1, 'low_wick_length'] = df['Open'] - df['Low']
    df['body_length'] = abs(df['Close'] - df['Open'])

    return df