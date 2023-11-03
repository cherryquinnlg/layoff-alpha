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
from single_stock import test_single_strategy
import pickle
import statsmodels.api as sm
import random
from sklearn.model_selection import ParameterGrid, train_test_split

# hpo part 1: randomized search

def res_for_all_tickers(df, max_layoff_percentage, max_layoff_no, holding_period, min_stock_price):
    f = (df.percentage_laid_off <= max_layoff_percentage) | (df.total_laid_off <= max_layoff_no)
    f = f & (df.country == "United States") & (df.adj_close >= min_stock_price)
    df = df[f]

    all_tickers = sorted(df.ticker.unique())

    SHARPE_RATIO = []
    RET = []
    DRAWDOWN = []
    PNL = []
    DF_RES = []

    for TICKER in all_tickers:
        fig, pnl, sharpe_ratio, ret, annual_ret, drawdown, df_res = test_single_strategy(TICKER, period=holding_period, verbose=False)

        SHARPE_RATIO.append(sharpe_ratio)
        RET.append(ret)
        DRAWDOWN.append(drawdown)
        PNL.append(pnl)
        DF_RES.append(df_res)

    df_summary = pd.DataFrame({"SHARPE_RATIO": SHARPE_RATIO,
                               "RET": RET,
                               "DRAWDOWN": DRAWDOWN,
                               "PNL": PNL})


    port_return = df_summary['RET'].mean()
    port_sharpe = df_summary['SHARPE_RATIO'].mean()
    port_drawdown = df_summary['DRAWDOWN'].mean()
    port_pnl = df_summary['PNL'].mean()

    return port_return, port_sharpe, port_drawdown, port_pnl

N = 3
param_grid = {'max_layoff_percentage': [0.05, 0.08, 0.10, 0.12, 0.15],
              'max_layoff_no': [200, 300, 500, 800, 1000, 1200, 1500, 2000, 15000],
              "holding_period": [7, 30, 60, 90, 120, 180, 360],
              "min_stock_price": [2, 5, 10, 20, 50, 80, 90, 100, 120, 150]}
all_perm = list(ParameterGrid(param_grid))
selected_perm = random.sample(all_perm, N)

# # for stratification
# bin_edges = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 1]
# bin_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# df["percentage_bin"] = pd.cut(df['percentage_laid_off'], bins=bin_edges, labels=bin_labels)
#
# bin_edges = [0, 100, 200, 500, 1000, 2000, 4000, 6000, 10000, 15000, 20000, 50000]
# bin_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# df["total_bin"] = pd.cut(df['total_laid_off'], bins=bin_edges, labels=bin_labels)
#
# bin_edges = [0, 2, 5, 10, 20, 50, 80, 90, 100, 120, 150, 300, 500, 1000]
# bin_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# df["close_bin"] = pd.cut(df['adj_close'], bins=bin_edges, labels=bin_labels)

df = pickle.load(open("oct_1.p", "rb"))
X_train, X_test, y_train, y_test = train_test_split(df, df["adj_close"], test_size=0.4, random_state=42)

for perm in selected_perm:
    max_layoff_percentage, max_layoff_no, holding_period, min_stock_price = perm["max_layoff_percentage"], \
        perm["max_layoff_no"], perm["holding_period"], perm["min_stock_price"]
    port_return, port_sharpe, port_drawdown, port_pnl = res_for_all_tickers(X_train, max_layoff_percentage, max_layoff_no, \
                                                                            holding_period, min_stock_price)
    perm["port_return"] = port_return

sorted_perm = sorted(selected_perm, key=lambda x: x['port_return'], reverse=True)
max_layoff_percentage, max_layoff_no, holding_period, min_stock_price = sorted_perm[0]["max_layoff_percentage"], \
        sorted_perm[0]["max_layoff_no"], sorted_perm[0]["holding_period"], sorted_perm[0]["min_stock_price"]

print("Best parameters:")
print(f"max_layoff_percentage: {max_layoff_percentage}, max_layoff_no: {max_layoff_no}, holding_period: {holding_period}, min_stock_price: {min_stock_price}")

i_port_return = sorted_perm[0]["port_return"]
o_port_return, o_port_sharpe, o_port_drawdown, o_port_pnl = res_for_all_tickers(X_test, max_layoff_percentage, max_layoff_no, \
                                                                            holding_period, min_stock_price)
print(f"In-sample portfolio return: {i_port_return}")
print(f"Out-of-sample portfolio return: {o_port_return}")


# hpo part 2, backtest portfolio buckets
#