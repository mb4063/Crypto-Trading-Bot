from dotenv import load_dotenv
import os

import pandas_ta as pan_ta
from finta import TA
from stockstats import StockDataFrame
import yfinance as yf

import datetime as dt
import random as rnd

from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.ticker as ticker
import mplfinance as mpf
from binance.client import Client
import pandas_ta as pan_ta
from datetime import datetime
from datetime import timedelta
import math
from time import sleep
import matplotlib.pyplot as plt

load_dotenv()

# Reading APIs from environment variables to keep them secure
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")


client = Client(api_key = api_key, api_secret = api_secret, tld = "com", testnet = False) # Mainnet!!!




## Kline, commonly known as Candlestick, which packs a lot of trade history information into a single data point.
## Fetching historical kline data from Binance API and converting it into a pandas DataFrame
def fetch_data(symbol, interval, start, end):
    bars = client.get_historical_klines(symbol = symbol, interval = interval, start_str = start, end_str = end)
    df = pd.DataFrame(bars)
    df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
    df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                "Clos Time", "Quote Asset Volume", "Number of Trades",
                "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df.set_index("Date", inplace = True)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors = "coerce")
    return df






## Calculating the simple moving averages of the closing prices
def calculate_SMAs(df: pd.DataFrame, short_window: int, long_window: int, period: int):
    # Create short and long length simple moving average column
    df[f'{short_window}_SMA'] = df['Close'].rolling(window = short_window, min_periods = period).mean()
    df[f'{long_window}_SMA'] = df['Close'].rolling(window = long_window, min_periods = period).mean()
    return df




def SMA_signals(df: pd.DataFrame, short_window: int, long_window: int):
    # create a new column 'Signal' such that if 20-length SMA is greater than 50-length SMA then set Signal as 1 else 0.

    df['Signal'] = 0.0  
    df['Signal'] = np.where(df[f'{short_window}_SMA'] > df[f'{long_window}_SMA'], 1.0, 0.0) 

    # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
    df['Position'] = df['Signal'].diff()
    # Note that Position = 1 indicates a 'buy' call and Position = -1 indicates 'sell' call
    
    df[df['Position'] == -1]
    #  Position = -1 indicates 'sell' call
    
    df[df['Position'] == 1]
    #  Position = 1 indicates a 'buy' call

    return df




