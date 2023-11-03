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
from helpers import *

class singleStockStrategy(backtrader.Strategy):
    def __init__(self, params):
        self.ticker = params["ticker"]
        self.layoff_dates =  [get_next_trading_day(d).date() for d in params["layoff_dates"]]
        self.clear_dates = [get_next_trading_day(d + timedelta(days=params["period"])).date() for d in params["layoff_dates"]]


    def next(self):
        if self.data.datetime.date(0) in self.layoff_dates:
            self.buy()
        elif self.data.datetime.date(0) in self.clear_dates:
            self.sell()

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            # b += 1

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))


class VolumeObserver(backtrader.Observer):
    lines = ('volume',)

    plotinfo = dict(plot=True, subplot=True)

    def next(self):
        self.lines.volume[0] = self.data.volume[0]

def test_single_strategy(TICKER):
    df = pickle.load(open("sept_29.p", "rb"))

    N = len(df)
    # SHARPE_RATIO = []
    # RET = []
    # DRAWDOWN = []
    # PNL = []

    STARTCASH = 100000
    LAYOFF_DATES = df.loc[df.ticker == TICKER, "date"].to_list()
    PERIOD = 100

    print(TICKER)
    params = {"ticker": TICKER, "layoff_dates": LAYOFF_DATES, "period": PERIOD}
    cerebro = backtrader.Cerebro()
    cerebro.addstrategy(singleStockStrategy, params)

    # add data
    data = backtrader.feeds.YahooFinanceCSVData(
        dataname=f"data/{TICKER}.csv",
        fromdate=datetime(2020,1,1))
    cerebro.adddata(data)

    # analysis
    # Add analyzers
    cerebro.addanalyzer(backtrader.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(backtrader.analyzers.SharpeRatio, _name='SharpeRatio')
    cerebro.addanalyzer(backtrader.analyzers.Returns, _name='Returns')
    cerebro.addanalyzer(backtrader.analyzers.DrawDown, _name='DrawDown')
    cerebro.addanalyzer(backtrader.analyzers.TimeReturn, _name='portfolio_value', timeframe=backtrader.TimeFrame.NoTimeFrame)

    # Add observers
    cerebro.addobserver(backtrader.observers.Broker)
    cerebro.addobserver(backtrader.observers.DrawDown)
    cerebro.addobserver(VolumeObserver)

    # add other parameters
    cerebro.addsizer(backtrader.sizers.PercentSizer, percents=90)
    cerebro.broker.setcommission(commission=0.002)
    cerebro.broker.setcash(STARTCASH)

    result = cerebro.run()

    # Printing all the summary statistics
    portvalue = cerebro.broker.getvalue()
    pnl = portvalue - STARTCASH

    print('Final Portfolio Value: ${}'.format(portvalue))
    print('P/L: ${}'.format(pnl))

    # p = cerebro.plot(iplot=False)
    # plt.show()
    #
    # for i, fig in enumerate(p):
    #     fig[0].set_size_inches(10, 10)
    #     fig[0].savefig(f'my_plot_1222.png', dpi=300)

    strat = result[0]

    pyfoliozer = strat.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

    sharpe_ratio = strat.analyzers.SharpeRatio.get_analysis()['sharperatio']
    ret = strat.analyzers.Returns.get_analysis()['rtot']
    drawdown = strat.analyzers.DrawDown.get_analysis()['max']['drawdown']

    print(f'Sharpe Ratio: {sharpe_ratio}')
    print(f'Return: {ret}')
    print(f"Annualized return: {ret * 365 / 100}")
    print(f'DrawDown: {drawdown}')

    print("\n")

    ################################################################################################
    # Portfolio Value
    portfolio_value = strat.analyzers.portfolio_value.get_analysis()

    # Volume, Cash, and Portfolio Value from observers
    volume = strat.observers.getbyname('Volume')[0].lines.volume.array
    cash = strat.observers.getbyname('Broker')[0].lines.cash.array
    portfolio_value_from_observer = strat.observers.getbyname('Broker')[0].lines.value.array

    # Time of trades
    times_of_trades = [order.executed.dt for order in strat.orders if order.status == order.Completed]

    df_res = pd.DataFrame(index=data.datetime.array)

    df_res['Volume'] = volume
    df_res['Cash'] = cash
    df_res['Portfolio Value'] = portfolio_value_from_observer
    df_res['Drawdown'] = drawdown['drawdown']
    df_res['Time of Trades'] = df.index.isin(times_of_trades)

    print(df_res)
    ################################################################################################

    p = 0

    return p, sharpe_ratio, ret, ret * 365 / 100, drawdown