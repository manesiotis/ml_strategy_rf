# metrics.py

import numpy as np
import pandas as pd

def calculate_cumulative_return(equity_curve: pd.Series):
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

def calculate_annualized_return(equity_curve: pd.Series, periods_per_year=252):
    total_return = calculate_cumulative_return(equity_curve)
    num_periods = len(equity_curve)
    return (1 + total_return) ** (periods_per_year / num_periods) - 1

def calculate_annualized_volatility(returns: pd.Series, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate=0.0, periods_per_year=252):
    excess_returns = returns - (risk_free_rate / periods_per_year)
    ann_vol = calculate_annualized_volatility(returns, periods_per_year)
    ann_return = returns.mean() * periods_per_year
    return ann_return / ann_vol if ann_vol != 0 else np.nan

def calculate_max_drawdown(equity_curve: pd.Series):
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    return drawdown.min()
