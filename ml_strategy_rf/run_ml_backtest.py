# run_ml_backtest.py

from data_loader import load_price_and_returns
from feature_engineering import create_features
from model import train_model_and_generate_signals
from backtester import Backtester
from metrics import (
    calculate_cumulative_return,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)
from plotting import plot_equity_curve

# === 1. Load data ===
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-12-31'

prices, returns = load_price_and_returns(ticker, start_date, end_date)

# === 2. Create features ===
features = create_features(prices[ticker])
features = features.dropna()
returns = returns.loc[features.index]


# === 3. Train model and generate signals ===
signals, model = train_model_and_generate_signals(features, returns)

# === 4. Run backtest ===
bt = Backtester(prices[ticker], signals)
results = bt.run_backtest()

# === 5. Print metrics ===
equity = results['Equity Curve']
strategy_returns = results['Strategy Return']

print(f"Cumulative Return:      {calculate_cumulative_return(equity):.2%}")
print(f"Annualized Return:      {calculate_annualized_return(equity):.2%}")
print(f"Annualized Volatility:  {calculate_annualized_volatility(strategy_returns):.2%}")
print(f"Sharpe Ratio:           {calculate_sharpe_ratio(strategy_returns):.2f}")
print(f"Max Drawdown:           {calculate_max_drawdown(equity):.2%}")
print("Signals summary:")
print(signals.value_counts())
print("Returns preview:")
print(returns.head())
# === 6. Plot equity curve ===
plot_equity_curve(
    equity_curve=equity,
    title=f"ML Strategy on {ticker}",
    save_path=f"plots/{ticker.lower()}_ml_equity.png"
)
