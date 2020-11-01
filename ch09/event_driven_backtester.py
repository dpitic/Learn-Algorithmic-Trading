"""Event driven backtester."""
import algolib.data as data
import matplotlib.pyplot as plt
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
        edbt.process_price_data(price_information['price'])
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
