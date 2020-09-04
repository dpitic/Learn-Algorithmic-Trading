"""Compare trading strategies by plotting profit and loss
This module compares the profit and loss of the basic and volatility adjusted
trading strategies for the mean reversion and trend following strategies.
"""
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Mean reversion trading strategies
    basic_mr = pd.read_csv('ch05/basic_mean_reversion.csv')
    vol_mr = pd.read_csv('ch05/volatility_mean_reversion.csv')

    plt.figure()
    plt.title('Mean Reversion Trading Strategy')
    basic_mr['PnL'].plot(x='Date', color='b', lw=1.0,
                         label='Basic Mean Reversion', legend=True, grid=True)
    vol_mr['PnL'].plot(x='Date', color='g', lw=1.0,
                       label='Volatility Adjusted Mean Reversion',
                       legend=True, grid=True)

    # Trend following trading strategies
    basic_tf = pd.read_csv('ch05/basic_trend_following.csv')
    vol_tf = pd.read_csv('ch05/volatility_trend_following.csv')

    plt.figure()
    plt.title('Trend Following Trading Strategy')
    basic_tf['PnL'].plot(x='Date', color='b', lw=1.0,
                         label='Basic Trend Following', legend=True, grid=True)
    vol_tf['PnL'].plot(x='Date', color='g', lw=1.0,
                       label='Volatility Adjusted Trend Following',
                       legend=True, grid=True)
    plt.show()


if __name__ == "__main__":
    main()
