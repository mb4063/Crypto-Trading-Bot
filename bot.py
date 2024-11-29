from dotenv import load_dotenv
import os

import pandas_ta as pan_ta
from finta import TA
from stockstats import StockDataFrame
import yfinance as yf
from tabulate import tabulate
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

## Variables for method testing
now = datetime.utcnow()
historical_days = 10
past = str(now - timedelta(days= historical_days))
symbol = "BTCUSDT"
bar_length = "1m"


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




df = fetch_data(symbol, bar_length, past, str(now))


df


## Calculating the simple moving averages of the closing prices
def calculate_SMAs(df: pd.DataFrame, short_window: int, long_window: int, period: int):
    # Create short and long length simple moving average column
    df[f'{short_window}_SMA'] = df['Close'].rolling(window = short_window, min_periods = period).mean()
    df[f'{long_window}_SMA'] = df['Close'].rolling(window = long_window, min_periods = period).mean()
    return df

dfa = calculate_SMAs(df, 20, 50, 1)


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

dfa = SMA_signals(dfa, 20, 50)

df_new = dfa[(dfa['Position'] == 1) | (dfa['Position'] == -1)]
df_new['Position'] = df_new['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
print(tabulate(df_new, headers = 'keys', tablefmt = 'psql'))

def computeRSI (data, time_window):
    diff = np.diff(data)
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]

    up_chg = pd.DataFrame(up_chg)
    down_chg = pd.DataFrame(down_chg)
    
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    rsi = int(rsi[0].iloc[-1])
    return rsi

def MACD(interval, symbol):
    klines2 = client.get_klines(symbol=symbol, interval=interval, limit='60')
    closeVal = [float(entry[4]) for entry in klines2]
    closeVal = pd.DataFrame(closeVal)
    ema12 = closeVal.ewm(span=12).mean()
    ema26 = closeVal.ewm(span=26).mean()
    macd = ema26 - ema12
    signal = macd.ewm(span=9).mean()

    macd = macd.values.tolist()
    signal = signal.values.tolist()
    
    if macd[-1] > signal[-1] and macd[-2] < signal[-2]:
        macdIndicator = 'BUY'
    elif macd[-1] < signal[-1] and macd[-2] > signal[-2]:
        macdIndicator = 'SELL'
    else:
        macdIndicator = 'HOLD'

    return macdIndicator

def stopLoss(symbol):
    today = DT.date.today()
    week_ago = today - DT.timedelta(days=6)
    week_ago = week_ago.strftime('%d %b, %Y')
    klines2 = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, str(week_ago))
    highVal = [float(entry[2]) for entry in klines2]
    lowVal = [float(entry[3]) for entry in klines2]
    closeVal = [float(entry[4]) for entry in klines2]
    avgDownDrop = (sum(highVal)/len(highVal)-sum(lowVal)/len(lowVal))/(sum(closeVal)/len(closeVal))
    stopVal = closeVal[-2]*(1-avgDownDrop)
    return stopVal

def takeprofit():
    winRate = 1.02
    return profit


