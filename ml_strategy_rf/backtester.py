import pandas as pd

class Backtester:
    def __init__(self, prices: pd.Series, signals: pd.Series, initial_cash: float = 10000):
        # Ensure input is Series
        self.prices = prices.squeeze()
        self.signals = signals.squeeze()
        self.initial_cash = initial_cash
        self.results = None

    def run_backtest(self):
        # Compute daily returns
        returns = self.prices.pct_change().fillna(0)

        # Apply strategy: multiply signal (shifted) by returns
        strategy_returns = (self.signals.shift(1).fillna(0) * returns).fillna(0)

        # Compute equity curve
        equity_curve = (1 + strategy_returns).cumprod() * self.initial_cash

        # Debug prints
        print("Strategy returns (first 5):")
        print(strategy_returns.head())

        print("Equity curve (first 5):")
        print(equity_curve.head())

        print("Any NaNs in strategy_returns?", strategy_returns.isna().sum())
        print("Any NaNs in equity_curve?", equity_curve.isna().sum())

        # Store results
        self.results = pd.DataFrame({
            'Price': self.prices,
            'Signal': self.signals,
            'Return': returns,
            'Strategy Return': strategy_returns,
            'Equity Curve': equity_curve
        })

        return self.results
