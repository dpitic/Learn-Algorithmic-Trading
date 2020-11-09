"""Event driven backtester."""
import matplotlib.pyplot as plt

import algolib.data as data
from algolib.backtest import EventDrivenBackTester


def main():
    # Load Google financial data into a DataFrame
    goog_data = data.get_google_data(start_date='2001-01-01')
    # Run event driven backtester
    edbt = EventDrivenBackTester()
    for line in zip(goog_data.index, goog_data['Adj Close']):
        date = line[0]
        price = line[1]
        price_information = {'date': date,
                             'price': float(price)}
        # Create orders from price (ticker) information
        edbt.create_orders(price_information['price'])
        # Run main trading loop
        edbt.process_events()

    # Plot output
    plt.plot(edbt.trading_strategy.list_paper_total,
             label='Paper Trading Using Event Driven Backtester')
    plt.plot(edbt.trading_strategy.list_total,
             label='Trading Using Event Driven Backtester')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
