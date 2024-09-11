import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

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