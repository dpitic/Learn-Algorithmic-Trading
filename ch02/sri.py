"""Support and Resistance Indicators"""
import matplotlib.pyplot as plt

from algolib.data import get_google_data


def main():
    # Google finance data file and start and end dates for data
    data_file = 'data/goog_data.pkl'
    start_date = '2014-01-01'
    end_date = '2018-01-01'
    # Load Google finance data
    goog_data = get_google_data(data_file, start_date, end_date)
    # To avoid complications with stock split, we only take dates without
    # splits. Therefore only keep 620 days.
    goog_data = goog_data.tail(620)
    lows = goog_data['Low']
    highs = goog_data['High']
    # Plot the data
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Google price in $')
    highs.plot(ax=ax1, color='c', lw=2.0)
    lows.plot(ax=ax1, color='y', lw=2.0)
    # Plot resistance limit
    plt.hlines(highs.head(200).max(), lows.index.values[0],
               lows.index.values[-1], linewidth=2, color='g',
               label='Resistance')
    # Plot support limit
    plt.hlines(lows.head(200).min(), lows.index.values[0],
               lows.index.values[-1], linewidth=2, color='r', label='Support')
    plt.axvline(x=lows.index.values[200],
                linewidth=2, color='b', linestyle=':')
    plt.grid()
    plt.legend()
    # Display plot and block
    plt.show()


if __name__ == "__main__":
    main()
