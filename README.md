# 🤖 Project 5: Machine Learning Strategy – Random Forests on Stock Returns

This project applies a supervised Machine Learning approach (Random Forest Classifier) to predict whether the next-day return of a stock will be positive. Based on these predictions, trading signals are generated and backtested on historical data.

---

## 📁 Project Structure

```
ml_strategy_rf/
│
├── data_loader.py             # Load historical prices & returns
├── feature_engineering.py     # Create lagged features, momentum, volatility, etc.
├── model.py                   # Train Random Forest and generate trading signals
├── backtester.py              # Core backtesting logic
├── metrics.py                 # Performance evaluation metrics
├── plotting.py                # Plot and save equity curve
├── run_ml_backtest.py         # Main script
│
├── plots/
│   └── aapl_ml_equity.png     # Output plot
│
└── README.md                  # Project description
```

---

## 🚀 How to Run

1. **Install required libraries**:

```bash
pip install pandas numpy matplotlib yfinance scikit-learn
```

2. **Run the script**:

```bash
python run_ml_backtest.py
```

This will:
- Download historical data (AAPL)
- Engineer ML features
- Train a Random Forest Classifier
- Generate signals and backtest the strategy
- Print performance metrics
- Save equity curve plot in `plots/`

---

## 📊 Features Used

- Lagged returns (lag_1 to lag_5)
- Momentum over 5 days
- 5-day and 10-day moving averages
- 5-day rolling volatility

---

## 🎯 Target Variable

The model predicts whether the **next-day return** will be positive:
- 1 → Buy (long)
- 0 → Cash (no position)

---

## 📈 Output Metrics (example)

```
Cumulative Return:      +65.40%
Annualized Return:      +13.25%
Annualized Volatility:  17.80%
Sharpe Ratio:           0.74
Max Drawdown:           -12.65%
```

---

## ✅ Notes

- Signals are generated from **ML predictions**
- Strategy is tested using a simple equity curve backtest
- No transaction costs, slippage, or position sizing applied

---

## 📌 Next Steps

- Try other models (XGBoost, Logistic Regression, etc.)
- Add more features (volume, RSI, etc.)
- Apply to other stocks or entire portfolios
- Incorporate transaction costs and realistic execution

---

## 👨‍💻 Author

Konstantinos Manesiotis  
Project 5 of Quant Trading Series
