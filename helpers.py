import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
import pytz
from cycler import cycler
import warnings
import pandas_market_calendars as mcal
import numpy as np
import seaborn as sns
import backtrader
import bt
from datetime import datetime, timedelta
from yahooquery import search
import pickle
import time
import os
from collections import defaultdict
import glob

nyse = mcal.get_calendar('NYSE')
trading_days = nyse.schedule(start_date='2020-01-01', end_date='2024-12-31')

def get_next_trading_day(date):
    for i in range(5):
        new_date = date + timedelta(days=i)
        if new_date in(trading_days.index):
            return new_date


def get_data(ticker):

    start_date = "2020-01-01"
    end_date = "2023-09-22"

    if ticker == "":
        return
    try:
        # # Many companies have multiple rounds of layoffs. Use fixed start & end dates instead
        # data = yf.download(ticker, start=start_date - pd.DateOffset(days=365), end=start_date + pd.DateOffset(days=365), actions=True)
        data = yf.download(ticker, start=start_date, end=end_date)
        data["log_close"] = np.log(data["Adj Close"])
        data["log_return"] = data["log_close"].diff()
        data.to_csv(f"data/{ticker}.csv")
    except:
        print(f"Cannot find data of {ticker}.")


def get_ticker(company_name):
    # Create a dictionary to store the company names and their corresponding ticker symbols
    time.sleep(1)

    try:
        result = search(company_name)
        if result and "quotes" in result:
            quotes = result["quotes"]
            if quotes:
                return quotes[0]["symbol"]

        print(f"Cannot find any info on {company_name}")

    except:
        print(f"Something is wrong with {company_name}.")

    return ""