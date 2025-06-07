# feature_engineering.py

import pandas as pd

def create_features(prices: pd.Series, lags=5):
    """
    Generates features from price data for ML model.

    Parameters:
    - prices: Series of adjusted close prices
    - lags: number of lagged returns to include

    Returns:
    - DataFrame of features
    """
    df = pd.DataFrame(index=prices.index)
    
    # Lagged returns
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = prices.pct_change(i)

    # Moving averages
    df['ma_5'] = prices.rolling(window=5).mean()
    df['ma_10'] = prices.rolling(window=10).mean()

    # Momentum
    df['momentum_5'] = prices / prices.shift(5) - 1

    # Volatility (rolling std of daily returns)
    daily_return = prices.pct_change()
    df['volatility_5'] = daily_return.rolling(window=5).std()

    # Drop initial NaNs
    df = df.dropna()

    return df
