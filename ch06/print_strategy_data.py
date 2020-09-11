"""Display the trading strategy data."""
import pandas as pd


def main():
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    # Load trading strategy results and data, and display on screen
    results = pd.read_csv('ch05/volatility_mean_reversion.csv')
    print(results)


if __name__ == "__main__":
    main()
