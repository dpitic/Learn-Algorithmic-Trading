"""Naive backtester."""
import algolib.data as data
import matplotlib.pyplot as plt
from algolib.backtest import ForLoopBackTester


def main():
    # Load Google financial data into a DataFrame
    goog_data = data.get_google_data(start_date='2001-01-01')
    # Run for loop backtester
    naive_backtester = ForLoopBackTester()
    for line in zip(goog_data.index, goog_data['Adj Close']):
        date = line[0]
        price = line[1]
        price_information = {'date': date,
                             'price': float(price)}
        is_tradable = naive_backtester.create_metrics(price_information)
        if is_tradable:
            naive_backtester.trade(price_information)

    # Plot output
    plt.plot(naive_backtester.list_total,
             label='Holdings+Cash using Naive Backtester')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
