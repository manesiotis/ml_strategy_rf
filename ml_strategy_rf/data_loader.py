# data_loader.py

import yfinance as yf
import pandas as pd

def load_price_and_returns(ticker, start_date, end_date):
    """
    Downloads adjusted close prices and computes daily returns.

    Parameters:
    - ticker: stock symbol (e.g., 'AAPL')
    - start_date: start date as 'YYYY-MM-DD'
    - end_date: end date as 'YYYY-MM-DD'

    Returns:
    - prices: DataFrame with adjusted close prices
    - returns: Series with daily returns
    """
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    prices = data[['Adj Close']].rename(columns={'Adj Close': ticker})
    returns = prices.pct_change().fillna(0)
    return prices, returns[ticker]
