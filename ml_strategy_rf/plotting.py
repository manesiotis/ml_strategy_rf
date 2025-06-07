# plotting.py

import matplotlib.pyplot as plt
import os

def plot_equity_curve(equity_curve, title="Equity Curve", save_path="plots/equity_curve.png"):
    """
    Plots and saves the equity curve to a PNG file.

    Parameters:
    - equity_curve: Series of portfolio values over time
    - title: Title of the plot
    - save_path: Where to save the PNG file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label='Equity Curve')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
